// Umi OS — 3次元の水のシミュレータ (MLS-MPM)   (C++ / WASM)
//
// webgpu-ocean (matsuoka-601) と同じ骨格を CPU で。
//
// なぜ MLS-MPM か
// ---------------
// SPH は近傍探索が高くつく（あちらも SPH をやめた理由がこれ）。
// MLS-MPM は粒子ごとに **3×3×3 の固定ステンシル**を触るだけの O(N) で、
// 近傍探索が要らない。分岐もリストも無いので、GPU が無くても素直に回る。
//
//   P2G  粒子の質量と運動量を、二次Bスプラインの重みで格子へ撒く。
//        アフィン項 C（APIC）と応力を同時に運ぶのが MLS-MPM の要点。
//   格子  v /= m、重力を足し、壁で速度を止める。
//   G2P  格子から速度を集め直し、同時に速度勾配 C を復元する。
//        体積比 J を J *= 1 + dt·tr(C) で更新する ＝ 圧縮の記録。
//
// 水（弱圧縮性流体）は応力が圧力だけなので
//        stress = -dt · p_vol · 4 · inv_dx² · E · (J − 1)
// のスカラーで済む。せん断を持たないぶん固体より安い。
//
// 参考: Hu et al. "A Moving Least Squares Material Point Method with
//       Displacement Discontinuity and Two-Way Rigid Body Coupling" (SIGGRAPH 2018)
#define OLIVEC_IMPLEMENTATION
#include "olive.c"
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define KEEP EMSCRIPTEN_KEEPALIVE
#else
#define KEEP
#endif

// ---------------------------------------------------------------- 画面
static const int FW = 960, FH = 600;

// ---------------------------------------------------------------- 格子
// 領域は (GX,GY,GZ)·dx。dx は一様（MPM の要請）。
// 水槽の実寸は 2.0 x 1.0 x 1.3 m に固定して、格子の細かさだけ変える。
// 細かくすると粒子は N^3 で増える（＝重くなる）が、水面のディテールが上がる。
static int   GX = 40, GY = 20, GZ = 26;
static float DX = 1.0f / 20.0f;
static float INV_DX = 20.0f;
static int   PERCELL = 2;                  // 1セルの1辺あたりの粒子数（2 なら 8個/セル）
static int   p_level = 1;                  // 粒子数プリセット（既定 = N=20）
static const int LEVEL_N[6] = { 16, 20, 24, 28, 32, 38 };   // 高さ方向のセル数

// 格子は (vx,vy,vz,m) を4つ並べて持つ。27セルを舐めるのでキャッシュに乗せたい。
static std::vector<float> grid;      // 4 * GX*GY*GZ
static inline int gidx(int i, int j, int k) { return ((k * GY + j) * GX + i) * 4; }

// ---------------------------------------------------------------- 粒子
// 64バイト＝1キャッシュライン に収める。
struct Par {
    float x, y, z;        // 位置
    float vx, vy, vz;     // 速度
    float C[9];           // アフィン速度勾配（APIC）
    float J;              // 体積比（1 が無圧縮）
};
static std::vector<Par> pars;

// ---------------------------------------------------------------- 物性
static float P_RHO   = 1.0f;
static float P_VOL   = 0.0f;      // dx/2 の立方体
static float P_MASS  = 0.0f;
static float E_BULK  = 400.0f;    // 体積弾性率（音速 sqrt(E/rho) が dt を決める）
static float GRAV    = 9.8f;
static float DT      = 6.0e-4f;   // 音速から決める（下の retune() 参照）
static int   SUBSTEP = 2;

// ★ dt は勝手に決めてはいけない。弾性波の速さ c=sqrt(E/rho) に対して
//   dt < dx/c を満たさないと必ず発散する（最初これで J が NaN になった）。
static void retune() {
    float c = sqrtf(E_BULK / P_RHO);
    DT = 0.35f * DX / c;
}

// 粒子数プリセット。水槽の実寸は変えず、格子の細かさだけ変える。
// ★ P_VOL を PERCELL に合わせて直すこと。1セルあたりの粒子数が変わるのに
//   1粒子の担う体積を固定にすると、密度が変わって水でなくなる。
static void set_level(int lv) {
    if (lv < 0) lv = 0; if (lv > 5) lv = 5;
    p_level = lv;
    int N = LEVEL_N[lv];
    DX = 1.0f / N; INV_DX = (float)N;
    GX = (int)(2.00f * N + 0.5f);
    GY = N;
    GZ = (int)(1.30f * N + 0.5f);
    retune();
}

// ---------------------------------------------------------------- 乱数
static uint32_t rng = 88675123u;
static inline float rnd() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return (rng & 0xFFFFFF) / (float)0x1000000; }
static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline uint32_t rgb(int r, int g, int b) {
    return 0xFF000000u | ((uint32_t)b << 16) | ((uint32_t)g << 8) | (uint32_t)r;
}

// ================================================================ 粒子の並べ替え
// ★CPU で MPM を回すときの一番大きい勝負どころ。
//   粒子は 3×3×3 のセルを触るので、配列上で近い粒子が空間的にも近ければ
//   その27セルはキャッシュに載ったままになる。シミュレーションが進むと
//   粒子の並びは空間的にバラバラになるので、ときどき詰め直す。
//   計数ソート（O(N + セル数)）なので、ほとんどタダ。
static std::vector<Par> parsTmp;
static std::vector<int> cellCount;
static void sort_particles() {
    const size_t N = pars.size();
    if (!N) return;
    const int NC = GX * GY * GZ;
    cellCount.assign(NC + 1, 0);
    auto cellOf = [&](const Par& q) {
        int i = (int)(q.x * INV_DX), j = (int)(q.y * INV_DX), k = (int)(q.z * INV_DX);
        i = i < 0 ? 0 : (i >= GX ? GX - 1 : i);
        j = j < 0 ? 0 : (j >= GY ? GY - 1 : j);
        k = k < 0 ? 0 : (k >= GZ ? GZ - 1 : k);
        return (k * GY + j) * GX + i;
    };
    for (size_t p = 0; p < N; ++p) cellCount[cellOf(pars[p]) + 1]++;
    for (int c = 0; c < NC; ++c) cellCount[c + 1] += cellCount[c];
    parsTmp.resize(N);
    for (size_t p = 0; p < N; ++p) parsTmp[cellCount[cellOf(pars[p])]++] = pars[p];
    pars.swap(parsTmp);
}

// ================================================================ MLS-MPM
static void step_once() {
    const size_t N = pars.size();
    std::fill(grid.begin(), grid.end(), 0.0f);

    const float invdx = INV_DX, dx = DX;
    const float stressK = -DT * P_VOL * 4.0f * invdx * invdx * E_BULK;

    // ---- P2G ----
    for (size_t p = 0; p < N; ++p) {
        Par& q = pars[p];
        float fxx = q.x * invdx - 0.5f, fyy = q.y * invdx - 0.5f, fzz = q.z * invdx - 0.5f;
        int bi = (int)floorf(fxx), bj = (int)floorf(fyy), bk = (int)floorf(fzz);
        if (bi < 0 || bj < 0 || bk < 0 || bi > GX - 3 || bj > GY - 3 || bk > GZ - 3) continue;
        float fx = q.x * invdx - bi, fy = q.y * invdx - bj, fz = q.z * invdx - bk;
        // 二次Bスプラインの重み
        float wx[3] = { 0.5f * (1.5f - fx) * (1.5f - fx), 0.75f - (fx - 1.0f) * (fx - 1.0f), 0.5f * (fx - 0.5f) * (fx - 0.5f) };
        float wy[3] = { 0.5f * (1.5f - fy) * (1.5f - fy), 0.75f - (fy - 1.0f) * (fy - 1.0f), 0.5f * (fy - 0.5f) * (fy - 0.5f) };
        float wz[3] = { 0.5f * (1.5f - fz) * (1.5f - fz), 0.75f - (fz - 1.0f) * (fz - 1.0f), 0.5f * (fz - 0.5f) * (fz - 0.5f) };

        float st = stressK * (q.J - 1.0f);          // 水はスカラー圧力だけ
        // affine = st*I + m*C
        float a0 = st + P_MASS * q.C[0], a1 = P_MASS * q.C[1], a2 = P_MASS * q.C[2];
        float a3 = P_MASS * q.C[3], a4 = st + P_MASS * q.C[4], a5 = P_MASS * q.C[5];
        float a6 = P_MASS * q.C[6], a7 = P_MASS * q.C[7], a8 = st + P_MASS * q.C[8];
        float mvx = P_MASS * q.vx, mvy = P_MASS * q.vy, mvz = P_MASS * q.vz;

        for (int k = 0; k < 3; ++k) {
            float dpz = (k - fz) * dx;
            for (int j = 0; j < 3; ++j) {
                float dpy = (j - fy) * dx;
                float wyz = wy[j] * wz[k];
                int base = gidx(bi, bj + j, bk + k);
                for (int i = 0; i < 3; ++i) {
                    float dpx = (i - fx) * dx;
                    float w = wx[i] * wyz;
                    float* g = &grid[base + i * 4];
                    g[0] += w * (mvx + a0 * dpx + a1 * dpy + a2 * dpz);
                    g[1] += w * (mvy + a3 * dpx + a4 * dpy + a5 * dpz);
                    g[2] += w * (mvz + a6 * dpx + a7 * dpy + a8 * dpz);
                    g[3] += w * P_MASS;
                }
            }
        }
    }

    // ---- 格子 ----
    const float gdt = GRAV * DT;
    for (int k = 0; k < GZ; ++k) for (int j = 0; j < GY; ++j) {
        float* g = &grid[gidx(0, j, k)];
        for (int i = 0; i < GX; ++i, g += 4) {
            float m = g[3];
            if (m <= 0.0f) { g[0] = g[1] = g[2] = 0.0f; continue; }
            float inv = 1.0f / m;
            g[0] *= inv; g[1] *= inv; g[2] *= inv;
            g[1] -= gdt;
            // 壁（すべりなし側は速度を止める）
            if (i < 3 && g[0] < 0) g[0] = 0;
            if (i > GX - 4 && g[0] > 0) g[0] = 0;
            if (j < 3 && g[1] < 0) g[1] = 0;
            if (j > GY - 4 && g[1] > 0) g[1] = 0;
            if (k < 3 && g[2] < 0) g[2] = 0;
            if (k > GZ - 4 && g[2] > 0) g[2] = 0;
        }
    }

    // ---- G2P ----
    for (size_t p = 0; p < N; ++p) {
        Par& q = pars[p];
        float fxx = q.x * invdx - 0.5f, fyy = q.y * invdx - 0.5f, fzz = q.z * invdx - 0.5f;
        int bi = (int)floorf(fxx), bj = (int)floorf(fyy), bk = (int)floorf(fzz);
        if (bi < 0 || bj < 0 || bk < 0 || bi > GX - 3 || bj > GY - 3 || bk > GZ - 3) continue;
        float fx = q.x * invdx - bi, fy = q.y * invdx - bj, fz = q.z * invdx - bk;
        float wx[3] = { 0.5f * (1.5f - fx) * (1.5f - fx), 0.75f - (fx - 1.0f) * (fx - 1.0f), 0.5f * (fx - 0.5f) * (fx - 0.5f) };
        float wy[3] = { 0.5f * (1.5f - fy) * (1.5f - fy), 0.75f - (fy - 1.0f) * (fy - 1.0f), 0.5f * (fy - 0.5f) * (fy - 0.5f) };
        float wz[3] = { 0.5f * (1.5f - fz) * (1.5f - fz), 0.75f - (fz - 1.0f) * (fz - 1.0f), 0.5f * (fz - 0.5f) * (fz - 0.5f) };

        float nvx = 0, nvy = 0, nvz = 0;
        float c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0, c8 = 0;
        for (int k = 0; k < 3; ++k) {
            float dpz = (k - fz);
            for (int j = 0; j < 3; ++j) {
                float dpy = (j - fy);
                float wyz = wy[j] * wz[k];
                int base = gidx(bi, bj + j, bk + k);
                for (int i = 0; i < 3; ++i) {
                    float dpx = (i - fx);
                    float w = wx[i] * wyz;
                    const float* g = &grid[base + i * 4];
                    float gx = g[0], gy = g[1], gz = g[2];
                    nvx += w * gx; nvy += w * gy; nvz += w * gz;
                    float ww = w * 4.0f * invdx;
                    c0 += ww * gx * dpx; c1 += ww * gx * dpy; c2 += ww * gx * dpz;
                    c3 += ww * gy * dpx; c4 += ww * gy * dpy; c5 += ww * gy * dpz;
                    c6 += ww * gz * dpx; c7 += ww * gz * dpy; c8 += ww * gz * dpz;
                }
            }
        }
        q.vx = nvx; q.vy = nvy; q.vz = nvz;
        q.C[0] = c0; q.C[1] = c1; q.C[2] = c2;
        q.C[3] = c3; q.C[4] = c4; q.C[5] = c5;
        q.C[6] = c6; q.C[7] = c7; q.C[8] = c8;
        q.J *= 1.0f + DT * (c0 + c4 + c8);          // 体積比の更新＝圧縮の記録
        q.x += DT * q.vx; q.y += DT * q.vy; q.z += DT * q.vz;
        // 念のため領域内に留める
        float lo = 2.0f * DX;
        q.x = clampf(q.x, lo, (GX - 3) * DX);
        q.y = clampf(q.y, lo, (GY - 3) * DX);
        q.z = clampf(q.z, lo, (GZ - 3) * DX);
    }
}

// ================================================================ 初期配置
static void fill_block(float x0, float y0, float z0, float x1, float y1, float z1, int perCell) {
    float s = DX / perCell;
    for (float z = z0; z < z1; z += s)
        for (float y = y0; y < y1; y += s)
            for (float x = x0; x < x1; x += s) {
                Par q{};
                q.x = x + (rnd() - 0.5f) * s * 0.6f;
                q.y = y + (rnd() - 0.5f) * s * 0.6f;
                q.z = z + (rnd() - 0.5f) * s * 0.6f;
                q.J = 1.0f;
                pars.push_back(q);
            }
}

static void setup(int scene) {
    pars.clear();
    float W = GX * DX, H = GY * DX, D = GZ * DX;
    // 本家 initDambreak と同じ取り方：幅は端から3セル、高さは 0.8H、奥行きは半分まで
    float e = 3.0f * DX;
    if (scene == 0) {
        fill_block(e, e, e, W - 4 * DX, H * 0.80f, D * 0.5f, PERCELL);
    } else {                    // 水槽：底に一様に溜める
        fill_block(e, e, e, W - 4 * DX, H * 0.34f, D - 4 * DX, PERCELL);
    }
}

// ================================================================ カメラ
// 本家: 距離 70 セル、fov 45°、注視点 (W/2, H/4, D/2)。camF = (FH/2)/tan(fov/2)
static float camAz = 0.34f, camEl = 0.13f, camR = 2.30f, camF = 724.0f;
static float camX, camY, camZ, cyaw, syaw, cpit, spit;
static float tgtX, tgtY, tgtZ;

static void setup_camera() {
    tgtX = GX * DX * 0.5f; tgtY = GY * DX * 0.25f; tgtZ = GZ * DX * 0.5f;
    camX = tgtX + camR * cosf(camEl) * sinf(camAz);
    camY = tgtY + camR * sinf(camEl);
    camZ = tgtZ - camR * cosf(camEl) * cosf(camAz);
    float yaw = -camAz;
    cyaw = cosf(yaw); syaw = sinf(yaw);
    cpit = cosf(camEl); spit = sinf(camEl);
}

static inline bool project(float x, float y, float z, float& sx, float& sy, float& depth) {
    float dx = x - camX, dy = y - camY, dz = z - camZ;
    float cx = cyaw * dx - syaw * dz;
    float cz = syaw * dx + cyaw * dz;
    float cy2 = cpit * dy + spit * cz;
    float cz2 = -spit * dy + cpit * cz;
    if (cz2 < 0.05f) return false;
    float inv = camF / cz2;
    sx = FW * 0.5f + cx * inv;
    sy = FH * 0.5f - cy2 * inv;
    depth = cz2;
    return true;
}

// ================================================================ 画面空間の流体レンダリング
// NVIDIA "Screen Space Fluid Rendering" (GDC 2010) の系譜。webgpu-ocean と同じ考え方を CPU で。
//
//   1) 深度   粒子を球のインポスタとして撒き、いちばん手前の視空間深度だけ残す。
//   2) 平滑化 その深度をぼかす。ただの Gauss だとシルエットが溶けるので、
//             中心深度から一定の窓に**切り詰めてから**混ぜる（narrow-range filter）。
//             双方向フィルタより輪郭の癖が少ない。
//   3) 法線   平滑化した深度から視空間の位置を復元し、差分の外積で法線を作る。
//             ★ここが肝で、粒子の球の法線ではなく「なめらかにした水面」の法線になる。
//   4) 厚み   深度テスト無しで加算した厚み。吸収（Beer-Lambert）に使う。
//   5) 着色   フレネル（Schlick）で反射と屈折を混ぜ、厚みぶん色を吸わせる。
//             背景は箱の中をレイキャストして作るので、屈折でちゃんと歪む。
static std::vector<uint32_t> px;
static std::vector<uint32_t> small_;   // RW×RH の流体
static std::vector<float> depthBuf;    // 視空間の深度（手前が小さい）。空は BIG
static std::vector<float> tmpBuf;
static std::vector<float> thickBuf;
static std::vector<float> bgBuf;       // 背景 RGB（低解像度、屈折で読む用）
static std::vector<uint32_t> bgFull;   // 背景（全解像度。画面の大半はこれで済む）
static bool bgDirty = true;
static int p_hud = 1;
static long frameNo = 0;
static const float BIG = 1e9f;
// ★流体は半解像度で描いて最後に拡大する。深度・平滑化・着色が一気に 1/4 になる。
//   画面空間の流体レンダリングはどのみち深度をぼかすので、半分でも見た目はほとんど変わらない。
static const int RS = 2;
static const int RW = FW / RS, RH = FH / RS;
static inline float rcamF() { return camF / RS; }

static float p_radius = 0.88f;         // 球の半径（dx 単位）
// 本家 fluid.wgsl の定数をそのまま
static const float BG_GREY = 0.80f;                                   // 背景の灰
static const float DIFF_R = 0.085f, DIFF_G = 0.6375f, DIFF_B = 0.9f;  // diffuseColor
static float p_absorb = 1.5f;          // density（本家の値）
static int   p_mode   = 0;             // 0:水 1:法線 2:深度 3:厚み 4:点
static float p_smooth = 2.0f;          // 平滑化の強さ

// ---- 速い exp とガンマ ----
// 毎画素で expf を3回・powf を4回呼ぶと、そこだけで 50ms を超える。表にする。
static const int EXPN = 2048;
static float expLut[EXPN + 1];
static const int GAMN = 1024;
static uint8_t gamLut[GAMN + 1];
static void build_luts() {
    for (int i = 0; i <= EXPN; ++i) expLut[i] = expf(-(float)i * (12.0f / EXPN));
    for (int i = 0; i <= GAMN; ++i) {
        float v = powf((float)i / GAMN, 0.85f) * 255.0f + 0.5f;
        gamLut[i] = (uint8_t)(v > 255 ? 255 : v);
    }
}
static inline float fexp(float negx) {          // negx <= 0 を想定
    float a = -negx;
    if (a >= 12.0f) return 0.0f;
    int i = (int)(a * (EXPN / 12.0f));
    return expLut[i];
}
static inline int fgam(float v) {               // [0,1] → 0..255（ガンマ0.85）
    if (v <= 0) return 0;
    if (v >= 1) return 255;
    return gamLut[(int)(v * GAMN)];
}

// 画素 → ワールド方向（背景のレイキャスト用）
static inline void ray_of(float sx, float sy, float& dx, float& dy, float& dz) {
    float a = (sx - RW * 0.5f) / rcamF();
    float b = -(sy - RH * 0.5f) / rcamF();
    float c = 1.0f;
    // 視空間 → ヨー後 → ワールド（project の逆）
    float wy = cpit * b - spit * c;
    float cz = spit * b + cpit * c;
    float wx = cyaw * a + syaw * cz;
    float wz = -syaw * a + cyaw * cz;
    float n = 1.0f / sqrtf(wx * wx + wy * wy + wz * wz);
    dx = wx * n; dy = wy * n; dz = wz * n;
}

// 背景：無地の灰色。水そのものを見せたいので、模様は置かない。
// ただし真っ平らだと屈折が「見えない」ので、床の接地面だけ淡く落として
// 水の底が少し歪んで見えるようにしてある。
static inline void bg_shade(float ox, float oy, float oz, float dx, float dy, float dz,
                            float& r, float& g, float& b) {
    // 本家と同じ、完全にフラットな灰 0.8
    (void)ox; (void)oy; (void)oz; (void)dx; (void)dy; (void)dz;
    r = BG_GREY; g = BG_GREY; b = BG_GREY;
}

// 空（反射に使う）
static inline void sky_shade(float dx, float dy, float dz, float& r, float& g, float& b) {
    // 本家は青空の cubemap。実際の平均色: 上(0.35,0.56,0.74) 下(0.24,0.35,0.49) 横(0.37,0.54,0.68)
    float t = clampf(dy * 0.5f + 0.5f, 0, 1);
    r = 0.235f + 0.185f * t;
    g = 0.350f + 0.245f * t;
    b = 0.485f + 0.290f * t;
    // 太陽（powf を使わず 8乗を掛け算で。毎画素なので効く）
    float d = dx * 0.42f + dy * 0.78f + dz * 0.46f;
    if (d > 0.90f) { float u = (d - 0.90f) * 10.0f; u *= u; u *= u; r += u * 2.2f; g += u * 2.2f; b += u * 2.1f; }
}

static void bake_bg() {
    bgBuf.assign((size_t)RW * RH * 3, 0.0f);
    for (int y = 0; y < RH; ++y) for (int x = 0; x < RW; ++x) {
        float dx, dy, dz; ray_of(x + 0.5f, y + 0.5f, dx, dy, dz);
        float r, g, b; bg_shade(camX, camY, camZ, dx, dy, dz, r, g, b);
        float* o = &bgBuf[((size_t)y * RW + x) * 3];
        o[0] = r; o[1] = g; o[2] = b;
    }
    // 全解像度の背景も焼く。画面の大半は水が無いので、そこはこれを貼るだけで済む。
    bgFull.assign((size_t)FW * FH, 0);
    for (int y = 0; y < FH; ++y) for (int x = 0; x < FW; ++x) {
        float a = (x + 0.5f - FW * 0.5f) / camF, b2 = -(y + 0.5f - FH * 0.5f) / camF, c = 1.0f;
        float wy = cpit * b2 - spit * c, cz = spit * b2 + cpit * c;
        float wx = cyaw * a + syaw * cz, wz = -syaw * a + cyaw * cz;
        float n = 1.0f / sqrtf(wx * wx + wy * wy + wz * wz);
        float r, g, b3; bg_shade(camX, camY, camZ, wx * n, wy * n, wz * n, r, g, b3);
        bgFull[(size_t)y * FW + x] = rgb(fgam(clampf(r, 0, 1)) , fgam(clampf(g, 0, 1)), fgam(clampf(b3, 0, 1)));
    }
    bgDirty = false;
}

// ---- 1) 深度パス：球のインポスタ ----
// 内部の粒子は表面に出ないので描かなくていい。
// 質量格子（P2G の副産物）で、6近傍が全部詰まっているセルの粒子を落とす。
// これで塗る画素が半分以下になる ── 球の面積が効くので一番効く間引き。
static inline bool is_interior(const Par& q) {
    int i = (int)(q.x * INV_DX), j = (int)(q.y * INV_DX), k = (int)(q.z * INV_DX);
    if (i < 1 || j < 1 || k < 1 || i >= GX - 1 || j >= GY - 1 || k >= GZ - 1) return false;
    const float TH = P_MASS * 3.0f;
    if (grid[gidx(i - 1, j, k) + 3] < TH) return false;
    if (grid[gidx(i + 1, j, k) + 3] < TH) return false;
    if (grid[gidx(i, j - 1, k) + 3] < TH) return false;
    if (grid[gidx(i, j + 1, k) + 3] < TH) return false;
    if (grid[gidx(i, j, k - 1) + 3] < TH) return false;
    if (grid[gidx(i, j, k + 1) + 3] < TH) return false;
    return true;
}

static long dbgDrawn = 0;
static int bbX0, bbX1, bbY0, bbY1;   // 水がある矩形（低解像度）
static void pass_depth() {
    std::fill(depthBuf.begin(), depthBuf.end(), BIG);
    std::fill(thickBuf.begin(), thickBuf.end(), 0.0f);
    float rw = p_radius * DX;
    dbgDrawn = 0;
    bbX0 = RW; bbX1 = -1; bbY0 = RH; bbY1 = -1;
    // 厚みは本家と同じ「粒子ごとに 0.05·√(1−r²) を足すだけ」の無次元量。
    // （物理的な光路長[m]に直す道もあるが、吸収係数まで本家に合わせたいのでこちらに揃える）
    const float tscale = 0.05f;
    for (const Par& q : pars) {
        if (is_interior(q)) continue;                    // ★内部は描かない
        float sx, sy, dep;
        if (!project(q.x, q.y, q.z, sx, sy, dep)) continue;
        sx /= RS; sy /= RS;
        dbgDrawn++;
        float rs = rcamF() * rw / dep;                   // 画面上の半径
        if (rs < 0.6f) rs = 0.6f;
        int x0 = (int)(sx - rs), x1 = (int)(sx + rs) + 1;
        int y0 = (int)(sy - rs), y1 = (int)(sy + rs) + 1;
        if (x1 < 0 || y1 < 0 || x0 >= RW || y0 >= RH) continue;
        if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
        if (x1 > RW - 1) x1 = RW - 1; if (y1 > RH - 1) y1 = RH - 1;
        if (x0 < bbX0) bbX0 = x0; if (x1 > bbX1) bbX1 = x1;
        if (y0 < bbY0) bbY0 = y0; if (y1 > bbY1) bbY1 = y1;
        float inv = 1.0f / (rs * rs);
        for (int y = y0; y <= y1; ++y) {
            float ddy = y + 0.5f - sy;
            for (int x = x0; x <= x1; ++x) {
                float ddx = x + 0.5f - sx;
                float d2 = (ddx * ddx + ddy * ddy) * inv;
                if (d2 >= 1.0f) continue;
                float nz = sqrtf(1.0f - d2);
                float z = dep - rw * nz;                 // 球の手前側
                size_t o = (size_t)y * RW + x;
                if (z < depthBuf[o]) depthBuf[o] = z;
                thickBuf[o] += nz * tscale;              // 本家と同じ積み方
            }
        }
    }
}

// ---- 2) 平滑化：narrow-range filter（分離型）----
static void filter_depth() {
    int R = (int)(5 * p_smooth); if (R < 1) R = 1; if (R > 18) R = 18;
    float sig = 0.55f * R;
    std::vector<float> w(R + 1);
    for (int i = 0; i <= R; ++i) w[i] = expf(-0.5f * (i * i) / (sig * sig));
    const float LO = 1.6f * p_radius * DX;               // 手前側は狭く（輪郭を溶かさない）
    const float HI = 5.0f * p_radius * DX;               // 奥側は広く
    int fy0 = bbY0 - R - 1, fy1 = bbY1 + R + 1, fx0 = bbX0 - R - 1, fx1 = bbX1 + R + 1;
    if (fy0 < 0) fy0 = 0; if (fx0 < 0) fx0 = 0;
    if (fy1 > RH - 1) fy1 = RH - 1; if (fx1 > RW - 1) fx1 = RW - 1;
    if (bbX1 < bbX0) return;
    for (int pass = 0; pass < 2; ++pass) {
        const int sxs = pass == 0 ? 1 : RW;
        for (int y = fy0; y <= fy1; ++y) for (int x = fx0; x <= fx1; ++x) {
            size_t o = (size_t)y * RW + x;
            float c = depthBuf[o];
            if (c >= BIG) { tmpBuf[o] = BIG; continue; }
            float lim = pass == 0 ? x : y;
            float ext = pass == 0 ? RW - 1 - x : RH - 1 - y;
            float sum = c * w[0], wsum = w[0];
            for (int t = 1; t <= R; ++t) {
                for (int s = -1; s <= 1; s += 2) {
                    if (s < 0 && t > lim) continue;
                    if (s > 0 && t > ext) continue;
                    float d = depthBuf[o + (size_t)(s * t * sxs)];
                    if (d >= BIG) d = c + HI;            // 空は「遠い」として窓の端に寄せる
                    d = clampf(d, c - LO, c + HI);       // ★ここが narrow-range
                    sum += w[t] * d; wsum += w[t];
                }
            }
            tmpBuf[o] = sum / wsum;
        }
        depthBuf.swap(tmpBuf);
    }
    // 厚みも軽くぼかす
    for (int pass = 0; pass < 2; ++pass) {
        const int sxs = pass == 0 ? 1 : RW;
        for (int y = 0; y < RH; ++y) for (int x = 0; x < RW; ++x) {
            size_t o = (size_t)y * RW + x;
            float lim = pass == 0 ? x : y, ext = pass == 0 ? RW - 1 - x : RH - 1 - y;
            float sum = thickBuf[o] * 2.0f, wsum = 2.0f;
            for (int t = 1; t <= 4; ++t) {
                if (t <= lim) { sum += thickBuf[o - (size_t)(t * sxs)]; wsum += 1.0f; }
                if (t <= ext) { sum += thickBuf[o + (size_t)(t * sxs)]; wsum += 1.0f; }
            }
            tmpBuf[o] = sum / wsum;
        }
        thickBuf.swap(tmpBuf);
    }
}

// 視空間の位置（深度から復元）
static inline void eye_pos(int x, int y, float z, float& ex, float& ey, float& ez) {
    ex = (x + 0.5f - RW * 0.5f) * z / rcamF();
    ey = -(y + 0.5f - RH * 0.5f) * z / rcamF();
    ez = z;
}

// ---- 3〜5) 法線・厚み・着色 ----
static void shade() {
    if (bgDirty) bake_bg();
    if (bbX1 < bbX0 || bbY1 < bbY0) return;
    // 拡大が双一次で1画素はみ出して読むので、着色は矩形より 3 画素広く塗る（継ぎ目防止）
    int sy0 = bbY0 - 3, sy1 = bbY1 + 3, sx0 = bbX0 - 3, sx1 = bbX1 + 3;
    if (sy0 < 0) sy0 = 0; if (sx0 < 0) sx0 = 0;
    if (sy1 > RH - 1) sy1 = RH - 1; if (sx1 > RW - 1) sx1 = RW - 1;
    for (int y = sy0; y <= sy1; ++y) {
        for (int x = sx0; x <= sx1; ++x) {
            size_t o = (size_t)y * RW + x;
            const float* bg = &bgBuf[o * 3];
            float z = depthBuf[o];
            if (z >= BIG) {
                small_[o] = rgb(fgam(bg[0]), fgam(bg[1]), fgam(bg[2]));   // ★外側と同じガンマで
                continue;
            }
            // --- 法線（深度の差分。端は内側差分に落とす）---
            float zl = (x > 0)      ? depthBuf[o - 1]  : z;
            float zr = (x < RW - 1) ? depthBuf[o + 1]  : z;
            float zu = (y > 0)      ? depthBuf[o - RW] : z;
            float zd = (y < RH - 1) ? depthBuf[o + RW] : z;
            if (zl >= BIG) zl = z; if (zr >= BIG) zr = z;
            if (zu >= BIG) zu = z; if (zd >= BIG) zd = z;
            float ex, ey, ez, ax, ay, az, bx, by, bz;
            eye_pos(x, y, z, ex, ey, ez);
            // 手前側の差分を選ぶ（シルエットで法線が寝ないように）
            float px1, py1, pz1, px2, py2, pz2;
            if (fabsf(zr - z) < fabsf(z - zl)) { eye_pos(x + 1, y, zr, px1, py1, pz1); ax = px1 - ex; ay = py1 - ey; az = pz1 - ez; }
            else { eye_pos(x - 1, y, zl, px1, py1, pz1); ax = ex - px1; ay = ey - py1; az = ez - pz1; }
            if (fabsf(zd - z) < fabsf(z - zu)) { eye_pos(x, y + 1, zd, px2, py2, pz2); bx = px2 - ex; by = py2 - ey; bz = pz2 - ez; }
            else { eye_pos(x, y - 1, zu, px2, py2, pz2); bx = ex - px2; by = ey - py2; bz = ez - pz2; }
            float nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
            float nl = sqrtf(nx * nx + ny * ny + nz * nz);
            if (nl < 1e-12f) { nx = 0; ny = 0; nz = -1; } else { nx /= nl; ny /= nl; nz /= nl; }
            if (nz > 0) { nx = -nx; ny = -ny; nz = -nz; }     // カメラを向かせる

            // 視空間法線 → ワールド
            float wy_ = cpit * ny - spit * nz;
            float cz_ = spit * ny + cpit * nz;
            float wx_ = cyaw * nx + syaw * cz_;
            float wz_ = -syaw * nx + cyaw * cz_;

            if (p_mode == 1) {
                small_[o] = rgb((int)((wx_ * 0.5f + 0.5f) * 255), (int)((wy_ * 0.5f + 0.5f) * 255), (int)((wz_ * 0.5f + 0.5f) * 255));
                continue;
            }
            if (p_mode == 2) {
                float t = clampf((z - camR * 0.4f) / (camR * 1.2f), 0, 1);
                int v = (int)((1 - t) * 255); small_[o] = rgb(v, v, v); continue;
            }
            if (p_mode == 3) {
                float t = clampf(thickBuf[o] * 3.0f, 0, 1);
                small_[o] = rgb((int)(t * 90), (int)(t * 180), (int)(t * 255)); continue;
            }

            // --- 視線ベクトル（ワールド）---
            float vx, vy, vz; ray_of(x + 0.5f, y + 0.5f, vx, vy, vz);

            // --- 屈折：本家は背景をずらさない。平らな背景に透過率を掛けるだけ ---
            float th = thickBuf[o];
            float tr = BG_GREY, tg = BG_GREY, tb2 = BG_GREY;

            // --- 吸収（Beer-Lambert）。水は赤から吸う ---
            // ★ ここが肝。透過率 exp(−density·厚み·(1−diffuseColor))。
            //   diffuseColor=(0.085,0.6375,0.9) なので消散係数は (0.915,0.3625,0.1)。
            //   **青はほとんど吸われない**（赤の 9 分の 1）。だから厚くなっても黒ではなく
            //   濃い群青に落ち着く。ここを (6.2,2.3,0.7) にしていたので黒くなっていた。
            float d = p_absorb * th;
            tr *= fexp(-d * (1.0f - DIFF_R));
            tg *= fexp(-d * (1.0f - DIFF_G));
            tb2 *= fexp(-d * (1.0f - DIFF_B));

            // --- 反射（空）---
            float dot = vx * wx_ + vy * wy_ + vz * wz_;
            float rxv = vx - 2 * dot * wx_, ryv = vy - 2 * dot * wy_, rzv = vz - 2 * dot * wz_;
            float sr, sg, sb; sky_shade(rxv, ryv, rzv, sr, sg, sb);

            // --- フレネル（Schlick）---
            float ct = clampf(-dot, 0, 1);
            float u = 1.0f - ct, u2 = u * u;
            float f = 0.02f + 0.98f * (u2 * u2 * u);      // Schlick の5乗は掛け算で足りる

            float r = tr * (1 - f) + sr * f;
            float g = tg * (1 - f) + sg * f;
            float b = tb2 * (1 - f) + sb * f;

            // --- 鏡面（太陽）---
            // 本家: lightDir はワールドの (0,0,-1)、H = normalize(lightDir − rayDir)、pow(dot(H,n), 250)
            float hx = 0.0f - vx, hy = 0.0f - vy, hz = -1.0f - vz;
            float hl = 1.0f / sqrtf(hx * hx + hy * hy + hz * hz + 1e-9f);
            float sp = clampf((hx * wx_ + hy * wy_ + hz * wz_) * hl, 0, 1);
            float s2 = sp * sp; s2 *= s2; s2 *= s2; s2 *= s2;    // sp^16
            float spec = s2 * s2 * s2 * s2;                      // sp^256 ≒ 本家の 250
            r += spec; g += spec; b += spec;

            // ★ 本家はトーンマップもガンマも掛けない。素通しでクランプするだけ。
            //   ここを掛けていたので全体が白っぽく眠くなっていた。
            small_[o] = rgb((int)(clampf(r, 0, 1) * 255.0f + 0.5f),
                            (int)(clampf(g, 0, 1) * 255.0f + 0.5f),
                            (int)(clampf(b, 0, 1) * 255.0f + 0.5f));
        }
    }
}

static void draw_points() {
    if (bgDirty) bake_bg();
    memcpy(px.data(), bgFull.data(), (size_t)FW * FH * 4);
    for (const Par& q : pars) {
        float sx, sy, dep;
        if (!project(q.x, q.y, q.z, sx, sy, dep)) continue;
        int ix = (int)sx, iy = (int)sy;
        if (ix < 0 || iy < 0 || ix >= FW || iy >= FH) continue;
        float s = sqrtf(q.vx * q.vx + q.vy * q.vy + q.vz * q.vz);
        float t = clampf(s / 4.0f, 0, 1);
        px[(size_t)iy * FW + ix] = rgb((int)(90 + 165 * t), (int)(150 + 105 * t), 235);
    }
}

// 低解像度の流体を画面へ拡大（双一次）
static void upscale() {
    // 背景をまるごと貼ってから、水のある矩形だけ拡大して上書きする。
    memcpy(px.data(), bgFull.data(), (size_t)FW * FH * 4);
    if (bbX1 < bbX0 || bbY1 < bbY0) return;
    int fx0 = bbX0 * RS - RS, fx1 = bbX1 * RS + RS;
    int fy0 = bbY0 * RS - RS, fy1 = bbY1 * RS + RS;
    if (fx0 < 0) fx0 = 0; if (fy0 < 0) fy0 = 0;
    if (fx1 > FW - 1) fx1 = FW - 1; if (fy1 > FH - 1) fy1 = FH - 1;
    for (int y = fy0; y <= fy1; ++y) {
        float fy = (y + 0.5f) / RS - 0.5f;
        int y0 = (int)floorf(fy); float ay = fy - y0;
        if (y0 < 0) { y0 = 0; ay = 0; }
        if (y0 > RH - 2) { y0 = RH - 2; ay = 1; }
        for (int x = fx0; x <= fx1; ++x) {
            float fx = (x + 0.5f) / RS - 0.5f;
            int x0 = (int)floorf(fx); float ax = fx - x0;
            if (x0 < 0) { x0 = 0; ax = 0; }
            if (x0 > RW - 2) { x0 = RW - 2; ax = 1; }
            const uint32_t* q = &small_[(size_t)y0 * RW + x0];
            uint32_t c00 = q[0], c01 = q[1], c10 = q[RW], c11 = q[RW + 1];
            float w00 = (1 - ax) * (1 - ay), w01 = ax * (1 - ay), w10 = (1 - ax) * ay, w11 = ax * ay;
            int r = (int)(((c00 & 255) * w00 + (c01 & 255) * w01 + (c10 & 255) * w10 + (c11 & 255) * w11));
            int g = (int)((((c00 >> 8) & 255) * w00 + ((c01 >> 8) & 255) * w01 + ((c10 >> 8) & 255) * w10 + ((c11 >> 8) & 255) * w11));
            int b = (int)((((c00 >> 16) & 255) * w00 + ((c01 >> 16) & 255) * w01 + ((c10 >> 16) & 255) * w10 + ((c11 >> 16) & 255) * w11));
            px[(size_t)y * FW + x] = rgb(r, g, b);
        }
    }
}

static double tSim = 0, tDepth = 0, tFilt = 0, tShade = 0;
static inline double nowms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
static void render_fluid() {
    if (p_mode == 4) { draw_points(); return; }
    double a = nowms(); pass_depth();
    double b = nowms(); filter_depth();
    double c = nowms(); shade(); upscale();
    tDepth += b - a; tFilt += c - b; tShade += nowms() - c;
}

// ================================================================ ABI
extern "C" {

KEEP void sim_reset();

KEEP int sim_w() { return FW; }
KEEP int sim_h() { return FH; }

KEEP void sim_reset() {
    grid.assign((size_t)GX * GY * GZ * 4, 0.0f);
    float sp = DX / PERCELL;                       // 粒子の間隔
    P_VOL = sp * sp * sp;                          // ★1粒子が担う体積は間隔の3乗
    P_MASS = P_VOL * P_RHO;
    retune();
    setup(0);
    setup_camera();
    bgDirty = true;
    frameNo = 0;
}

KEEP int sim_init(int seed, int) {
    if (seed == 0) {
        uint64_t t = (uint64_t)std::chrono::system_clock::now().time_since_epoch().count();
        seed = (int)(t ^ (t >> 32));
    }
    rng = (uint32_t)seed | 1u;
    for (int i = 0; i < 8; ++i) rnd();
    build_luts();
    px.assign((size_t)FW * FH, 0);
    small_.assign((size_t)RW * RH, 0);
    depthBuf.assign((size_t)RW * RH, BIG);
    tmpBuf.assign((size_t)RW * RH, 0.0f);
    thickBuf.assign((size_t)RW * RH, 0.0f);
    bgDirty = true;
    sim_reset();
    return 1;
}

KEEP void sim_set(int id, double v) {
    switch (id) {
    case 0: E_BULK = (float)v; retune(); break;
    case 1: GRAV = (float)v; break;
    case 2: SUBSTEP = (int)(v + 0.5); break;
    case 3: camAz = (float)v * 0.0174532925f; setup_camera(); bgDirty = true; break;
    case 4: camEl = (float)v * 0.0174532925f; setup_camera(); bgDirty = true; break;
    case 5: camR = (float)v; setup_camera(); bgDirty = true; break;
    case 6: p_radius = (float)v; break;
    case 7: p_absorb = (float)v; break;
    case 8: p_smooth = (float)v; break;
    case 9: p_mode = (int)(v + 0.5); break;
    case 10: { int lv = (int)(v + 0.5); if (lv != p_level) { set_level(lv); sim_reset(); } break; }
    case 11: { int pc = (int)(v + 0.5); if (pc != PERCELL && pc >= 1 && pc <= 3) { PERCELL = pc; sim_reset(); } break; }
    case 13: p_hud = v > 0.5 ? 1 : 0; break;
    default: break;
    }
}

KEEP void sim_action(int id) {
    if (id == 0) sim_reset();
    else if (id == 1) { pars.clear(); setup(1); }
}

KEEP double sim_get(int id) {
    switch (id) {
    case 0: return (double)pars.size();
    case 1: return (double)frameNo;
    case 2: return (double)p_level;
    case 3: return (double)PERCELL;
    case 4: return (double)GX;
    case 5: return (double)GY;
    case 6: return (double)GZ;
    default: return 0;
    }
}

KEEP void sim_click(double, double) {}

KEEP void sim_step(int n) {
    double a = nowms();
    for (int f = 0; f < n; ++f) {
        if ((frameNo & 7) == 0) sort_particles();      // 8フレームに1回で足りる
        for (int s = 0; s < SUBSTEP; ++s) step_once();
        frameNo++;
    }
    tSim += nowms() - a;
    render_fluid();
}

KEEP uint32_t* sim_render() {
    if (p_hud) {
        char buf[160];
        Olivec_Canvas oc = olivec_canvas(px.data(), FW, FH, FW);
        snprintf(buf, sizeof buf, "UMI  MLS-MPM  particles %zu  grid %dx%dx%d  substeps %d",
                 pars.size(), GX, GY, GZ, SUBSTEP);
        olivec_text(oc, buf, 14, 12, olivec_default_font, 2, rgb(150, 200, 235));
    }
    return px.data();
}

}  // extern "C"

// ================================================================ ネイティブ自己テスト
#ifndef __EMSCRIPTEN__
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdio>
int main(int argc, char** argv) {
    int frames = argc > 1 ? atoi(argv[1]) : 200;
    const char* out = argc > 2 ? argv[2] : "umi.png";
    sim_init(12345, 0);
    if (const char* e = getenv("LEVEL")) { sim_set(10, atof(e)); }
    if (const char* e = getenv("PERCELL")) { sim_set(11, atof(e)); }
    if (const char* e = getenv("SUB")) { sim_set(2, atof(e)); }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < frames; ++i) sim_step(1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / frames;
    sim_render();
    std::vector<uint8_t> b((size_t)FW * FH * 3);
    for (int i = 0; i < FW * FH; ++i) {
        uint32_t c = px[i];
        b[(size_t)i * 3 + 0] = c & 0xFF; b[(size_t)i * 3 + 1] = (c >> 8) & 0xFF; b[(size_t)i * 3 + 2] = (c >> 16) & 0xFF;
    }
    stbi_write_png(out, FW, FH, 3, b.data(), FW * 3);
    double js = 0, jmin = 1e9, jmax = -1e9;
    for (const Par& q : pars) { js += q.J; if (q.J < jmin) jmin = q.J; if (q.J > jmax) jmax = q.J; }
    double tmax = 0, tsum = 0; int tn = 0;
    for (size_t i2 = 0; i2 < thickBuf.size(); ++i2) if (thickBuf[i2] > 1e-6f) { if (thickBuf[i2] > tmax) tmax = thickBuf[i2]; tsum += thickBuf[i2]; tn++; }
    printf("%s  particles=%zu  %.1f ms/frame (%d substeps)\n", out, pars.size(), ms, SUBSTEP);
    printf("     J: mean=%.4f min=%.4f max=%.4f | thickness max=%.3f m mean=%.3f m | tank %.2fx%.2fx%.2f m\n",
           js / pars.size(), jmin, jmax, tmax, tn ? tsum / tn : 0.0, GX * DX, GY * DX, GZ * DX);
    printf("     naiwake/frame:  MPM %.1f ms   depth %.1f ms   filter %.1f ms   shade %.1f ms\n",
           tSim / frames, tDepth / frames, tFilt / frames, tShade / frames);
    return 0;
}
#endif
