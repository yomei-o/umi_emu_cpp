// Umi OS — 3次元の水のシミュレータ (MLS-MPM)   (C++ / WASM)
//
// webgpu-ocean (matsuoka-601) を **GPU を使わず C++ だけで** 焼き直したもの。
//
// ★2回目の写経。1回目は「MLS-MPM の教科書どおり」に書いたせいで本家と別物になった。
//   本家の mls-mpm/*.wgsl と render/*.wgsl を読み直して、定数まで揃えたのがこの版。
//   何が違っていたかは resume.md の「踏んだ罠」に全部書いてある。要点だけ:
//
//   1) 単位。本家は **格子1マス = 1** で計算する（dx=1、m でも s でもない）。
//      重力 0.3、dt 0.20、音速 √(3·5/4)≈1.9。**流速が音速を超える**（マッハ2）ほど柔らかい。
//      教科書どおりの「弱圧縮性の水」(E=400, マッハ0.2) にすると、水が飴のように粘って
//      1フレームがほとんど進まない。海に見えない原因の半分はこれだった。
//   2) 圧力。本家は **格子から測った密度の Tait 状態方程式** で、しかも
//      `max(0, …)` で **負圧を捨てる**。教科書の J（体積比）方式は負圧＝引力を持つので、
//      水が自分を丸めて塊になる。「球体に見える」原因はこれ。
//   3) 粘性。本家は stress に `0.1·(C + Cᵀ)` を足している。無いと表面が毛羽立つ。
//   4) 箱の形。本家は 40×30×60 ＝ **奥に長い水槽**。こちらは 40×20×26 の浅い平鍋だった。
//      奥行きが幅の1.5倍あって、45°から見下ろすから「海面の広がり」が出る。
//   5) 厚み（吸収）は **内部の粒子も全部足す**。手前の粒子だけで足すと薄くなる。
//
// MLS-MPM の骨格（本家の compute pass と1対1）
//   clearGrid → P2G1（質量と運動量）→ P2G2（密度→圧力→応力）→ updateGrid（v/=m, 重力, 壁）
//   → G2P（速度と C を集め直し、位置を進め、壁のバネ）      これを1フレームに2回。
//
// 参考: Hu et al. "A Moving Least Squares Material Point Method ..." (SIGGRAPH 2018)
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
// ★単位は「格子1マス」。dx = 1。位置も速度も半径もカメラ距離も、全部マス数で持つ。
//   本家 mls-mpm がそうなっているので、定数をそのまま持ってこられる。
static int GX = 35, GY = 25, GZ = 55;

// 壁の位置は別に持つ。奥行きだけ動かせるようにして「波を作る」に使う（本家の Box width スライダ）。
static float realX = 35.0f, realY = 25.0f, realZ = 55.0f;

// 格子は (vx,vy,vz,m) を4つ並べて持つ。27マスを舐めるのでキャッシュに乗せたい。
// i が連続。P2G/G2P の内側ループも i にしてある。
static std::vector<float> grid;
static inline int gidx(int i, int j, int k) { return ((k * GY + j) * GX + i) * 4; }

// ---------------------------------------------------------------- 粒子
// 64バイト＝1キャッシュライン。質量は 1 固定（本家と同じ）。
struct Par {
    float x, y, z;        // 位置（マス単位）
    float vx, vy, vz;     // 速度
    float C[9];           // アフィン速度勾配（APIC）。行優先: C[3*row+col]
    float dens;           // 格子から測った密度（P2G2 の副産物。自己テストの健全性チェック用）
};
static std::vector<Par> pars;

// ---------------------------------------------------------------- 物性
// ★本家 mls-mpm.ts の constants をそのまま。
static float STIFFNESS = 3.0f;      // Tait の k
static float REST_DENS = 4.0f;      // 1マスあたりの粒子数（間隔 0.65 なら 1/0.65³ = 3.64）
static float DYN_VISC  = 0.1f;      // 動粘性
static float DT        = 0.20f;     // 固定。CFL は音速 1.9 × 0.2 = 0.39 マスで足りている
static float GRAV      = 0.3f;      // マス/時間²
static int   SUBSTEP   = 2;         // 本家 execute() の for (i<2)
static const float SPACING = 0.65f; // 初期配置の粒子間隔

// 粒子数プリセット。本家 main.ts の 4 段（[0] と [1] がそれ）に、CPU 用の軽い段を足した。
static const int   LV_BOX[5][3] = { {30,22,45}, {35,25,55}, {40,30,60}, {45,40,80}, {50,50,80} };
static const int   LV_NUM[5]    = { 20000, 40000, 70000, 120000, 200000 };
static const float LV_DIST[5]   = { 52.0f, 60.0f, 70.0f, 90.0f, 100.0f };
static int p_level = 1;

// ---------------------------------------------------------------- 乱数
static uint32_t rng = 88675123u;
static inline float rnd() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return (rng & 0xFFFFFF) / (float)0x1000000; }
static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline uint32_t rgb(int r, int g, int b) {
    return 0xFF000000u | ((uint32_t)b << 16) | ((uint32_t)g << 8) | (uint32_t)r;
}

// ================================================================ 粒子の並べ替え
// CPU で MPM を回すときの勝負どころ。粒子は 3×3×3 マスを触るので、配列上で近い粒子が
// 空間的にも近ければその27マスはキャッシュに載ったままになる。計数ソートなのでほぼタダ。
static std::vector<Par> parsTmp;
static std::vector<int> cellCount;
static void sort_particles() {
    const size_t N = pars.size();
    if (!N) return;
    const int NC = GX * GY * GZ;
    cellCount.assign(NC + 1, 0);
    auto cellOf = [&](const Par& q) {
        int i = (int)q.x, j = (int)q.y, k = (int)q.z;
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
// ★格子を (vx,vy,vz,m) の4つ組で持っているので、**1節点 = 1ベクタ** にちょうど乗る。
//   27節点のループが load / fma / store 各1回になる。ここが CPU 版の一番の伸びしろだった。
//   grid は std::vector<float> なので 16 バイト境界を保証できない → aligned(4) で受ける。
typedef float v4  __attribute__((vector_size(16)));
typedef float v4u __attribute__((vector_size(16), aligned(4)));
static inline v4 splat(float a) { return (v4){ a, a, a, a }; }

// 二次Bスプラインの重み。本家と同じ取り方:
//   セル番号 c = floor(pos)、中心からのずれ d = pos − (c + 0.5) ∈ [−0.5, 0.5]
//   節点は c−1, c, c+1 のセル中心（c+0.5 など）
static inline void bspline(float p, int& c, float w[3], float& d) {
    c = (int)floorf(p);
    d = p - (c + 0.5f);
    w[0] = 0.5f * (0.5f - d) * (0.5f - d);
    w[1] = 0.75f - d * d;
    w[2] = 0.5f * (0.5f + d) * (0.5f + d);
}

static void step_once() {
    const size_t N = pars.size();
    std::fill(grid.begin(), grid.end(), 0.0f);

    // ---- P2G 1: 質量と運動量（APIC の Q = C·(節点 − 粒子) を載せる）----
    for (size_t p = 0; p < N; ++p) {
        Par& q = pars[p];
        int ci, cj, ck; float wx[3], wy[3], wz[3], dx_, dy_, dz_;
        bspline(q.x, ci, wx, dx_); bspline(q.y, cj, wy, dy_); bspline(q.z, ck, wz, dz_);
        if (ci < 1 || cj < 1 || ck < 1 || ci > GX - 2 || cj > GY - 2 || ck > GZ - 2) continue;
        const float* C = q.C;
        // 4番目の車線に 1 を置いておくと、そのまま質量になる（w·1 = w）
        const v4 vp = (v4){ q.vx, q.vy, q.vz, 1.0f };
        // Q = C·dist を「軸ごとの列ベクトル × 距離」に分解しておく
        const v4 cx = (v4){ C[0], C[3], C[6], 0.0f };
        const v4 cy = (v4){ C[1], C[4], C[7], 0.0f };
        const v4 cz = (v4){ C[2], C[5], C[8], 0.0f };
        v4 cxd[3];
        for (int gx = 0; gx < 3; ++gx) cxd[gx] = cx * splat((float)(gx - 1) - dx_);
        for (int gz = 0; gz < 3; ++gz) {
            v4 acc_z = cz * splat((float)(gz - 1) - dz_);
            for (int gy = 0; gy < 3; ++gy) {
                v4 base = vp + acc_z + cy * splat((float)(gy - 1) - dy_);
                float wyz = wy[gy] * wz[gz];
                v4u* g = (v4u*)&grid[gidx(ci - 1, cj - 1 + gy, ck - 1 + gz)];
                g[0] = (v4)g[0] + splat(wx[0] * wyz) * (base + cxd[0]);
                g[1] = (v4)g[1] + splat(wx[1] * wyz) * (base + cxd[1]);
                g[2] = (v4)g[2] + splat(wx[2] * wyz) * (base + cxd[2]);
            }
        }
    }

    // ---- P2G 2: 密度 → 圧力（Tait）→ 応力 → 運動量 ----
    // ★ここが本家の心臓。J（体積比）は使わない。**格子から測った密度**で圧力を出し、
    //   max(0,…) で負圧を捨てる。だから水は引き合わない ＝ 塊に丸まらない。
    for (size_t p = 0; p < N; ++p) {
        Par& q = pars[p];
        int ci, cj, ck; float wx[3], wy[3], wz[3], dx_, dy_, dz_;
        bspline(q.x, ci, wx, dx_); bspline(q.y, cj, wy, dy_); bspline(q.z, ck, wz, dz_);
        if (ci < 1 || cj < 1 || ck < 1 || ci > GX - 2 || cj > GY - 2 || ck > GZ - 2) continue;

        float dens = 0.0f;
        for (int gz = 0; gz < 3; ++gz)
            for (int gy = 0; gy < 3; ++gy) {
                float wyz = wy[gy] * wz[gz];
                const float* g = &grid[gidx(ci - 1, cj - 1 + gy, ck - 1 + gz)];
                dens += wyz * (wx[0] * g[3] + wx[1] * g[7] + wx[2] * g[11]);
            }
        q.dens = dens;
        if (dens <= 1e-6f) continue;

        float volume = 1.0f / dens;
        float r = dens / REST_DENS, r2 = r * r;
        float press = STIFFNESS * (r2 * r2 * r - 1.0f);            // Tait, 指数 5
        if (press < 0.0f) press = 0.0f;                            // ★負圧を捨てる

        // stress = −p·I + μ·(C + Cᵀ)   （対称なので6成分）
        const float* C = q.C;
        float s00 = -press + DYN_VISC * (C[0] + C[0]);
        float s11 = -press + DYN_VISC * (C[4] + C[4]);
        float s22 = -press + DYN_VISC * (C[8] + C[8]);
        float s01 = DYN_VISC * (C[1] + C[3]);
        float s02 = DYN_VISC * (C[2] + C[6]);
        float s12 = DYN_VISC * (C[5] + C[7]);

        float f = -volume * 4.0f * DT;
        // eq_16_term0 = −V·4·stress·dt。対称なので3本の列ベクトルで足りる（4番目の車線は 0）
        const v4 tx = (v4){ f * s00, f * s01, f * s02, 0.0f };
        const v4 ty = (v4){ f * s01, f * s11, f * s12, 0.0f };
        const v4 tz = (v4){ f * s02, f * s12, f * s22, 0.0f };
        v4 txd[3];
        for (int gx = 0; gx < 3; ++gx) txd[gx] = tx * splat((float)(gx - 1) - dx_);
        for (int gz = 0; gz < 3; ++gz) {
            v4 acc_z = tz * splat((float)(gz - 1) - dz_);
            for (int gy = 0; gy < 3; ++gy) {
                v4 base = acc_z + ty * splat((float)(gy - 1) - dy_);
                float wyz = wy[gy] * wz[gz];
                v4u* g = (v4u*)&grid[gidx(ci - 1, cj - 1 + gy, ck - 1 + gz)];
                g[0] = (v4)g[0] + splat(wx[0] * wyz) * (base + txd[0]);
                g[1] = (v4)g[1] + splat(wx[1] * wyz) * (base + txd[1]);
                g[2] = (v4)g[2] + splat(wx[2] * wyz) * (base + txd[2]);
            }
        }
    }

    // ---- updateGrid: v /= m、重力、壁で速度成分を落とす ----
    const float gdt = GRAV * DT;
    const int bx = (int)ceilf(realX) - 3, by = (int)ceilf(realY) - 3, bz = (int)ceilf(realZ) - 3;
    for (int k = 0; k < GZ; ++k) for (int j = 0; j < GY; ++j) {
        float* g = &grid[gidx(0, j, k)];
        bool wy_ = (j < 2 || j > by), wz_ = (k < 2 || k > bz);
        for (int i = 0; i < GX; ++i, g += 4) {
            float m = g[3];
            if (m <= 0.0f) { g[0] = g[1] = g[2] = 0.0f; continue; }
            float inv = 1.0f / m;
            g[0] *= inv; g[1] = g[1] * inv - gdt; g[2] *= inv;
            if (i < 2 || i > bx) g[0] = 0.0f;
            if (wy_)             g[1] = 0.0f;
            if (wz_)             g[2] = 0.0f;
        }
    }

    // ---- G2P: 速度と C を集め直し、位置を進め、壁のバネを足す ----
    const float wallMinV = 3.0f;
    const float wallMaxX = realX - 4.0f, wallMaxY = realY - 4.0f, wallMaxZ = realZ - 4.0f;
    for (size_t p = 0; p < N; ++p) {
        Par& q = pars[p];
        int ci, cj, ck; float wx[3], wy[3], wz[3], dx_, dy_, dz_;
        bspline(q.x, ci, wx, dx_); bspline(q.y, cj, wy, dy_); bspline(q.z, ck, wz, dz_);
        if (ci < 1 || cj < 1 || ck < 1 || ci > GX - 2 || cj > GY - 2 || ck > GZ - 2) continue;

        // v は3成分ぶん、B は「速度 ⊗ 距離」なので距離の軸ごとに3本の蓄積に分かれる。
        v4 vs = splat(0.0f), bx = splat(0.0f), by = splat(0.0f), bz = splat(0.0f);
        v4 dxs[3];
        for (int gx = 0; gx < 3; ++gx) dxs[gx] = splat((float)(gx - 1) - dx_);
        for (int gz = 0; gz < 3; ++gz) {
            v4 dzs = splat((float)(gz - 1) - dz_);
            for (int gy = 0; gy < 3; ++gy) {
                v4 dys = splat((float)(gy - 1) - dy_);
                float wyz = wy[gy] * wz[gz];
                const v4u* g = (const v4u*)&grid[gidx(ci - 1, cj - 1 + gy, ck - 1 + gz)];
                for (int gx = 0; gx < 3; ++gx) {
                    v4 vw = splat(wx[gx] * wyz) * (v4)g[gx];
                    vs += vw;
                    bx += vw * dxs[gx]; by += vw * dys; bz += vw * dzs;
                }
            }
        }
        q.vx = vs[0]; q.vy = vs[1]; q.vz = vs[2];
        q.C[0] = 4 * bx[0]; q.C[1] = 4 * by[0]; q.C[2] = 4 * bz[0];
        q.C[3] = 4 * bx[1]; q.C[4] = 4 * by[1]; q.C[5] = 4 * bz[1];
        q.C[6] = 4 * bx[2]; q.C[7] = 4 * by[2]; q.C[8] = 4 * bz[2];

        q.x += DT * q.vx; q.y += DT * q.vy; q.z += DT * q.vz;
        q.x = clampf(q.x, 1.0f, realX - 2.0f);
        q.y = clampf(q.y, 1.0f, realY - 2.0f);
        q.z = clampf(q.z, 1.0f, realZ - 2.0f);

        // 壁のバネ。**次の位置を予測して**押し返す（本家 g2p.wgsl の k=3, stiffness=0.3）。
        // 格子だけで壁を作ると角で溜まるので、粒子側にも柔らかい壁を置いている。
        const float k = 3.0f, ws = 0.3f;
        float xn = q.x + q.vx * DT * k, yn = q.y + q.vy * DT * k, zn = q.z + q.vz * DT * k;
        if (xn < wallMinV) q.vx += ws * (wallMinV - xn);
        if (xn > wallMaxX) q.vx += ws * (wallMaxX - xn);
        if (yn < wallMinV) q.vy += ws * (wallMinV - yn);
        if (yn > wallMaxY) q.vy += ws * (wallMaxY - yn);
        if (zn < wallMinV) q.vz += ws * (wallMinV - zn);
        if (zn > wallMaxZ) q.vz += ws * (wallMaxZ - zn);
    }
}

// ================================================================ 初期配置
// ★本家 initDambreak をそのまま。
//   x は端から3マス、y は 0 から 0.8·H、z は手前から奥行きの半分まで。間隔 0.65。
//   ジッタは **3成分に同じ値**（0〜2マス）を足す。本家がそうなっている（対角にずれる）。
//   下の層から詰めていって、目標粒子数で打ち切る。
static int targetN = 40000;
static void setup_dambreak() {
    pars.clear();
    pars.reserve(targetN);
    for (float j = 0; j < GY * 0.80f && (int)pars.size() < targetN; j += SPACING)
        for (float i = 3; i < GX - 4 && (int)pars.size() < targetN; i += SPACING)
            for (float k = 3; k < GZ * 0.5f && (int)pars.size() < targetN; k += SPACING) {
                float jit = 2.0f * rnd();
                Par q{};
                q.x = i + jit; q.y = j + jit; q.z = k + jit;
                pars.push_back(q);
            }
}

// 水槽：底に一様に溜める（落ち着いた海面を見たいとき用）
static void settle(int substeps);
static void setup_tank() {
    pars.clear();
    pars.reserve(targetN);
    float h = GY * 0.30f;
    for (float j = 1; j < h && (int)pars.size() < targetN; j += SPACING)
        for (float i = 3; i < GX - 4 && (int)pars.size() < targetN; i += SPACING)
            for (float k = 3; k < GZ - 4 && (int)pars.size() < targetN; k += SPACING) {
                float jit = SPACING * rnd();
                Par q{};
                q.x = i + jit; q.y = j + jit; q.z = k + jit;
                pars.push_back(q);
            }
    settle(70);
}

// 初期配置を落ち着かせる。間隔 0.65 の格子は密度 3.64（rest 4）で少し足りないので、
// 置いただけだと自重で沈んで壁を叩く。粘性を上げて数十ステップ回し、速度を捨てる。
// ★静かな水面から始めたい場面（水槽・飛沫）はこれを通す。
static void settle(int substeps) {
    float keep = DYN_VISC;
    DYN_VISC = 1.5f;
    for (int i = 0; i < substeps; ++i) step_once();
    DYN_VISC = keep;
    for (Par& q : pars) {
        q.vx = q.vy = q.vz = 0.0f;
        for (int i = 0; i < 9; ++i) q.C[i] = 0.0f;
    }
}

// 飛沫：浅く張った水に水塊を落とす。本家の写真の「中央の大きな飛沫」がこれで出る。
static void setup_splash() {
    pars.clear();
    pars.reserve(targetN);
    const int ballN = targetN / 12;
    const int poolN = targetN - ballN;
    // 浅く敷く
    float h = GY * 0.32f;
    for (float j = 1; j < h && (int)pars.size() < poolN; j += SPACING)
        for (float i = 3; i < GX - 4 && (int)pars.size() < poolN; i += SPACING)
            for (float k = 3; k < GZ - 4 && (int)pars.size() < poolN; k += SPACING) {
                float jit = SPACING * rnd();
                Par q{}; q.x = i + jit; q.y = j + jit; q.z = k + jit;
                pars.push_back(q);
            }
    settle(70);                       // ★先に水面を落ち着かせる。でないと着水より前に壁を叩く
    // 中身まで詰めた球を上から落とす。半径は個数から引く（体積 = 個数 × 間隔³）
    float vol = (float)ballN * SPACING * SPACING * SPACING;
    float rad = cbrtf(vol * 3.0f / (4.0f * 3.14159265f));
    float cx = GX * 0.5f, cy = GY - rad - 2.0f, cz = GZ * 0.5f;
    for (float z = cz - rad; z <= cz + rad; z += SPACING)
        for (float y = cy - rad; y <= cy + rad; y += SPACING)
            for (float x = cx - rad; x <= cx + rad; x += SPACING) {
                float d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
                if (d2 > rad * rad) continue;
                Par q{}; q.x = x; q.y = y; q.z = z; q.vy = -10.0f;
                pars.push_back(q);
            }
}

// ================================================================ カメラ
// 本家 main.ts / camera.ts:  fov 45°、距離 70 マス、注視点 (W/2, H/4, D/2)、
//   方位 π/4 = 45°、仰角 −π/12 = 15° 見下ろし。camF = (FH/2)/tan(fov/2)
// ★この「距離が箱の幅の 1.75 倍」「奥行きが幅の 1.5 倍」「45°から見下ろす」の3つで
//   海面の広がりが出る。前の版は距離 1.15 倍・奥行き 0.65 倍・方位 20° だったので
//   画面いっぱいの塊に見えていた。
static float camAz = 135.0f * 0.0174532925f;   // ワールドの +z 側から見る（本家と同じ向き）
static float camEl = 15.0f * 0.0174532925f;
static float camR  = 60.0f;
static const float TAN_HALF_FOV = 0.41421356f; // tan(22.5°)
static const float camF = (FH * 0.5f) / TAN_HALF_FOV;
static float camX, camY, camZ, cyaw, syaw, cpit, spit;
static float tgtX, tgtY, tgtZ;

static void setup_camera() {
    tgtX = GX * 0.5f; tgtY = GY * 0.25f; tgtZ = GZ * 0.5f;
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
    if (cz2 < 1.0f) return false;
    float inv = camF / cz2;
    sx = FW * 0.5f + cx * inv;
    sy = FH * 0.5f - cy2 * inv;
    depth = cz2;
    return true;
}

// ================================================================ 画面空間の流体レンダリング
// NVIDIA "Screen Space Fluid Rendering" (GDC 2010) の系譜。本家 render/*.wgsl と同じ手順・同じ定数。
//
//   1) 深度   粒子を球のインポスタとして撒き、いちばん手前の視空間深度だけ残す。
//   2) 平滑化 双方向フィルタ（bilateral）。**分離型を4往復**。1往復だと粒がそのまま残る。
//             窓の幅は深度に反比例（遠い粒子は画面上で小さいので窓も狭い）。
//   3) 法線   平滑化した深度から視空間の位置を復元し、差分の外積で法線を作る。
//   4) 厚み   深度テスト無しで **全粒子** を加算。σ≈8px で大きくぼかす。吸収に使う。
//   5) 着色   フレネル（Schlick, F0=0.02）で反射と屈折を混ぜ、厚みぶん色を吸わせる。
static std::vector<uint32_t> px;
static std::vector<uint32_t> small_;   // RW×RH の流体
static std::vector<float> depthBuf;    // 視空間の深度（手前が小さい）。空は BIG
static std::vector<float> tmpBuf;
static std::vector<float> thickBuf;    // TW×THH（粗い）
static std::vector<float> thickTmp;
static bool bgDirty = true;
static int p_hud = 1;
static long frameNo = 0;
static const float BIG = 1e9f;

// 深度と着色は 1/RS、厚みは 1/TS。厚みは σ≈8px でぼかすので粗くて足りる。
// RS は実行時に変えられる（1 = 全解像度。きれいだが4倍重い）。バッファは RS=1 の分だけ確保する。
static int RS = 2;
static int RW = FW / 2, RH = FH / 2;
static const int TS = 4;
static const int TW = FW / TS, THH = FH / TS;
static inline float rcamF() { return camF / RS; }
static void set_rs(int rs) {
    if (rs < 1) rs = 1; if (rs > 4) rs = 4;
    RS = rs; RW = FW / RS; RH = FH / RS;
}

// ---- 本家 fluid.wgsl / fluidRender.ts の定数 ----
static float p_radius  = 0.60f;        // 球の半径（マス）。本家 mlsmpmRadius
static const float BG_GREY = 0.80f;                                   // bgColor
static const float DIFF_R = 0.085f, DIFF_G = 0.6375f, DIFF_B = 0.9f;  // diffuseColor
static float p_absorb  = 1.5f;         // density
static int   p_mode    = 0;            // 0:水 1:法線 2:深度 3:厚み 4:点
static float p_smooth  = 1.0f;         // 平滑化の窓の倍率（1.0 = 本家どおり）
static const float TALPHA = 0.05f;     // particle_alpha
static const int   FILT_ITER = 4;      // 本家 fluidRender.ts の for (iter<4)
static const int   BLUR_SCALE = 10;    // blurdDepthScale
static const int   BLUR_FILTER_SIZE = 12;

// ---- 速い exp ----
// 毎画素で expf を3回呼ぶと、そこだけで数十ms かかる。表にする。
static const int EXPN = 2048;
static float expLut[EXPN + 1];
// 球のインポスタは毎画素 sqrt(1−r²) を要る。ここも表にする（厚みは後で σ=8px でぼかすので粗くて足りる）。
static const int SQN = 1024;
static float sqLut[SQN + 1];
static void build_luts() {
    for (int i = 0; i <= EXPN; ++i) expLut[i] = expf(-(float)i * (12.0f / EXPN));
    for (int i = 0; i <= SQN; ++i) sqLut[i] = sqrtf(1.0f - (float)i / SQN);
}
static inline float fsq1m(float d2) { return sqLut[(int)(d2 * SQN)]; }   // sqrt(1-d2), 0<=d2<1
static inline float fexp(float negx) {          // negx <= 0 を想定
    float a = -negx;
    if (a >= 12.0f) return 0.0f;
    return expLut[(int)(a * (EXPN / 12.0f))];
}

// 画素 → ワールド方向
static inline void ray_of(float sx, float sy, float& dx, float& dy, float& dz) {
    float a = (sx - RW * 0.5f) / rcamF();
    float b = -(sy - RH * 0.5f) / rcamF();
    float c = 1.0f;
    float wy = cpit * b - spit * c;
    float cz = spit * b + cpit * c;
    float wx = cyaw * a + syaw * cz;
    float wz = -syaw * a + cyaw * cz;
    float n = 1.0f / sqrtf(wx * wx + wy * wy + wz * wz);
    dx = wx * n; dy = wy * n; dz = wz * n;
}

// 空（反射に使う）。
// ★本家の cubemap は青空ではなく **"Industrial Sunset 02 (Pure Sky)" (polyhaven, CC0)** ＝
//   海の上の夕暮れ。上は青、地平は白く霞み、下は鋼色の水面、そして地平の一方向に橙の太陽。
//   本家の絵に出ている「水面を走る暖色の筋」はこれの映り込み。青空の cubemap だと出ない。
//   ここは写真を持たずに、その6面の色を手続きで作る。
static const float SUNX = 0.520f, SUNY = 0.060f, SUNZ = -0.852f;   // 太陽の方向（低い）
static inline void sky_shade(float dx, float dy, float dz, float& r, float& g, float& b) {
    float y = clampf(dy, -1.0f, 1.0f);
    if (y >= 0.0f) {                       // 空：上は青、地平は白く霞む
        float u = 1.0f - y; u = u * u * u;
        r = 0.33f + 0.47f * u; g = 0.57f + 0.27f * u; b = 0.78f + 0.10f * u;
    } else {                               // 水面：地平は明るく、真下は鋼色
        float u = 1.0f + y; u = u * u * u;
        r = 0.17f + 0.55f * u; g = 0.28f + 0.48f * u; b = 0.42f + 0.38f * u;
    }
    // 夕日。地平の帯 × 太陽への近さ で暖色を足す
    float hz = 1.0f - (y < 0 ? -y : y); hz *= hz; hz *= hz;
    float sd = dx * SUNX + dy * SUNY + dz * SUNZ;
    if (sd > 0.0f) {
        float sw = sd * sd; sw *= sw;                       // sd^4
        r += 0.60f * sw * hz; g += 0.32f * sw * hz; b += 0.02f * sw * hz;
        if (sd > 0.9995f) { r += 1.0f; g += 0.9f; b += 0.6f; }   // 太陽そのもの
    }
}

// 背景は本家と同じ、完全にフラットな灰 0.8。★ガンマもトーンマップも掛けない。
static std::vector<uint32_t> bgFull;
static void bake_bg() {
    uint32_t c = rgb((int)(BG_GREY * 255 + 0.5f), (int)(BG_GREY * 255 + 0.5f), (int)(BG_GREY * 255 + 0.5f));
    bgFull.assign((size_t)FW * FH, c);
    bgDirty = false;
}

// ---- 1) 深度＋厚みパス：球のインポスタ ----
// ★内部の粒子も **厚みには必ず足す**（本家は深度テスト無しで全粒子を加算している）。
//   前の版は内部を捨てていたので、厚い所が薄く出て色が合わなかった。
//   深度のほうは間引いてよいが、本家のカメラ距離（箱の1.75倍）だと球が画面上で3px程度しか
//   ないので、全部撒いても安い。間引きの穴で表面が割れるほうが害が大きい。
static long dbgDrawn = 0;
static int bbX0, bbX1, bbY0, bbY1;   // 水がある矩形（1/RS 座標）
static void pass_depth() {
    std::fill(depthBuf.begin(), depthBuf.begin() + (size_t)RW * RH, BIG);
    std::fill(thickBuf.begin(), thickBuf.end(), 0.0f);
    const float rw = p_radius;
    const float fD = camF / RS, fT = camF / TS;
    dbgDrawn = 0;
    bbX0 = RW; bbX1 = -1; bbY0 = RH; bbY1 = -1;
    // ★手前から撒く。粒子はセル順（k が外側）に並んでいるので、カメラが +z 側にいるときは
    //   配列を逆から舐めれば手前→奥になる。上の早期打ち切りはこの順のときに一番効く。
    const int NP = (int)pars.size();
    int i0 = 0, i1 = NP, istep = 1;
    if (camZ > tgtZ) { i0 = NP - 1; i1 = -1; istep = -1; }
    for (int ip = i0; ip != i1; ip += istep) {
        const Par& q = pars[ip];
        float sx, sy, dep;
        if (!project(q.x, q.y, q.z, sx, sy, dep)) continue;
        dbgDrawn++;

        // --- 深度（1/RS）---
        {
            float cx = sx / RS, cy = sy / RS;
            float rs = fD * rw / dep;
            if (rs < 0.6f) rs = 0.6f;
            int x0 = (int)(cx - rs), x1 = (int)(cx + rs) + 1;
            int y0 = (int)(cy - rs), y1 = (int)(cy + rs) + 1;
            if (!(x1 < 0 || y1 < 0 || x0 >= RW || y0 >= RH)) {
                if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
                if (x1 > RW - 1) x1 = RW - 1; if (y1 > RH - 1) y1 = RH - 1;
                if (x0 < bbX0) bbX0 = x0; if (x1 > bbX1) bbX1 = x1;
                if (y0 < bbY0) bbY0 = y0; if (y1 > bbY1) bbY1 = y1;
                float inv = 1.0f / (rs * rs);
                const float znear = dep - rw;      // この球が書ける最も手前の値
                for (int y = y0; y <= y1; ++y) {
                    float ddy = y + 0.5f - cy;
                    float* d = &depthBuf[(size_t)y * RW];
                    for (int x = x0; x <= x1; ++x) {
                        // ★内部の粒子はここで全部落ちる。球のいちばん手前でも既存より奥なら、
                        //   その画素は絶対に更新されない。sqrt に入る前に切るのが効く。
                        //   （前の版は「6近傍が詰まった粒子を捨てる」間引きをしていたが、
                        //     表面の粒子まで落ちて水面に穴が空いた。こちらは厳密で速い）
                        if (znear >= d[x]) continue;
                        float ddx = x + 0.5f - cx;
                        float d2 = (ddx * ddx + ddy * ddy) * inv;
                        if (d2 >= 1.0f) continue;
                        float z = dep - rw * fsq1m(d2);              // 球の手前側
                        if (z < d[x]) d[x] = z;
                    }
                }
            }
        }

        // --- 厚み（1/TS、全粒子、深度テスト無しで加算）---
        {
            float cx = sx / TS, cy = sy / TS;
            float rs = fT * rw / dep;
            float amp = 1.0f;
            if (rs < 0.8f) { float s = rs / 0.8f; amp = s * s; rs = 0.8f; }  // 面積を保存する
            int x0 = (int)(cx - rs), x1 = (int)(cx + rs) + 1;
            int y0 = (int)(cy - rs), y1 = (int)(cy + rs) + 1;
            if (x1 < 0 || y1 < 0 || x0 >= TW || y0 >= THH) continue;
            if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
            if (x1 > TW - 1) x1 = TW - 1; if (y1 > THH - 1) y1 = THH - 1;
            float inv = 1.0f / (rs * rs), a = amp * TALPHA;
            for (int y = y0; y <= y1; ++y) {
                float ddy = y + 0.5f - cy;
                float* t = &thickBuf[(size_t)y * TW];
                for (int x = x0; x <= x1; ++x) {
                    float ddx = x + 0.5f - cx;
                    float d2 = (ddx * ddx + ddy * ddy) * inv;
                    if (d2 >= 1.0f) continue;
                    t[x] += a * fsq1m(d2);
                }
            }
        }
    }
}

// ---- 2) 平滑化 ----
// 本家 bilateral.wgsl をそのまま。窓の幅は
//   filter_size = min(100, ceil(K / depth)),  K = 12 · 直径 · 0.05 · (画面高/2) / tan(fov/2)
// ★4往復する（本家 fluidRender.ts）。1往復だと粒がそのまま見える。ここを1回にしていた。
static int filt_size_max = 0;
static void filter_depth() {
    if (bbX1 < bbX0) return;
    const float Kfs = (float)BLUR_FILTER_SIZE * (2.0f * p_radius) * 0.05f
                      * ((float)RH * 0.5f) / TAN_HALF_FOV * p_smooth;
    const int FSCAP = 24;                                  // CPU なので上限を切る（本家は 100）
    const float sigd = (p_radius * BLUR_SCALE) / 3.0f;     // depth_threshold / 3
    const float two_sigd = 2.0f * sigd * sigd;
    // 空間側の重みは窓幅ごとに同じものを何万回も使うので、先に表にする。
    static float wsp[FSCAP + 1][FSCAP + 1];
    for (int fs = 1; fs <= FSCAP; ++fs) {
        float sig = (float)fs / 3.0f, inv2 = 1.0f / (2.0f * sig * sig);
        for (int t = 0; t <= fs; ++t) wsp[fs][t] = expf(-(float)(t * t) * inv2);
    }

    int fy0 = bbY0 - 1, fy1 = bbY1 + 1, fx0 = bbX0 - 1, fx1 = bbX1 + 1;
    if (fy0 < 0) fy0 = 0; if (fx0 < 0) fx0 = 0;
    if (fy1 > RH - 1) fy1 = RH - 1; if (fx1 > RW - 1) fx1 = RW - 1;

    filt_size_max = 0;
    for (int it = 0; it < FILT_ITER; ++it) {
        for (int pass = 0; pass < 2; ++pass) {
            const int stride = pass == 0 ? 1 : RW;
            for (int y = fy0; y <= fy1; ++y) {
                for (int x = fx0; x <= fx1; ++x) {
                    size_t o = (size_t)y * RW + x;
                    float c = depthBuf[o];
                    if (c >= BIG) { tmpBuf[o] = BIG; continue; }   // 背景は背景のまま（輪郭を広げない）
                    int fs = (int)ceilf(Kfs / c);
                    if (fs < 1) fs = 1; if (fs > FSCAP) fs = FSCAP;
                    if (fs > filt_size_max) filt_size_max = fs;
                    const float* wr = wsp[fs];
                    int lim = pass == 0 ? x : y;
                    int ext = pass == 0 ? RW - 1 - x : RH - 1 - y;
                    float sum = c, wsum = 1.0f;
                    for (int t = 1; t <= fs; ++t) {
                        float w = wr[t];
                        for (int s = -1; s <= 1; s += 2) {
                            if (s < 0 ? t > lim : t > ext) continue;
                            float d = depthBuf[o + (size_t)(s * t * stride)];
                            if (d >= BIG) continue;                // 空は重み 0（本家も実質そう）
                            float rd = d - c;
                            float wd = w * fexp(-rd * rd / two_sigd);
                            sum += d * wd; wsum += wd;
                        }
                    }
                    tmpBuf[o] = sum / wsum;
                }
            }
            depthBuf.swap(tmpBuf);
        }
    }
}

// 厚みのぼかし。本家 gaussian.wgsl は 30px 窓・σ=10px（画面高 ≈756）。
// 画面高 600・1/TS で焼くので窓 6px・σ=2px に相当する。
static void filter_thick() {
    const int fs = 6;
    const float sig = (float)fs / 3.0f, inv_two_sig = 1.0f / (2.0f * sig * sig);
    float w[16];
    for (int i = 0; i <= fs; ++i) w[i] = expf(-(float)(i * i) * inv_two_sig);
    for (int pass = 0; pass < 2; ++pass) {
        const int stride = pass == 0 ? 1 : TW;
        for (int y = 0; y < THH; ++y) for (int x = 0; x < TW; ++x) {
            size_t o = (size_t)y * TW + x;
            int lim = pass == 0 ? x : y;
            int ext = pass == 0 ? TW - 1 - x : THH - 1 - y;
            float sum = thickBuf[o], wsum = 1.0f;
            for (int t = 1; t <= fs; ++t) {
                if (t <= lim) { sum += w[t] * thickBuf[o - (size_t)(t * stride)]; wsum += w[t]; }
                if (t <= ext) { sum += w[t] * thickBuf[o + (size_t)(t * stride)]; wsum += w[t]; }
            }
            thickTmp[o] = sum / wsum;
        }
        thickBuf.swap(thickTmp);
    }
}

// 厚みを 1/RS 座標で双一次に読む
static inline float thick_at(int x, int y) {
    float fx = ((x + 0.5f) * RS) / TS - 0.5f, fy = ((y + 0.5f) * RS) / TS - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float ax = fx - x0, ay = fy - y0;
    if (x0 < 0) { x0 = 0; ax = 0; } if (x0 > TW - 2) { x0 = TW - 2; ax = 1; }
    if (y0 < 0) { y0 = 0; ay = 0; } if (y0 > THH - 2) { y0 = THH - 2; ay = 1; }
    const float* t = &thickBuf[(size_t)y0 * TW + x0];
    return t[0] * (1 - ax) * (1 - ay) + t[1] * ax * (1 - ay)
         + t[TW] * (1 - ax) * ay + t[TW + 1] * ax * ay;
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
    // 拡大が双一次で1画素はみ出して読むので、着色は矩形より3画素広く塗る（継ぎ目防止）
    int sy0 = bbY0 - 3, sy1 = bbY1 + 3, sx0 = bbX0 - 3, sx1 = bbX1 + 3;
    if (sy0 < 0) sy0 = 0; if (sx0 < 0) sx0 = 0;
    if (sy1 > RH - 1) sy1 = RH - 1; if (sx1 > RW - 1) sx1 = RW - 1;
    const uint32_t bgpix = rgb((int)(BG_GREY * 255 + 0.5f), (int)(BG_GREY * 255 + 0.5f), (int)(BG_GREY * 255 + 0.5f));
    for (int y = sy0; y <= sy1; ++y) {
        for (int x = sx0; x <= sx1; ++x) {
            size_t o = (size_t)y * RW + x;
            float z = depthBuf[o];
            if (z >= BIG) { small_[o] = bgpix; continue; }

            // --- 法線（深度の差分。手前側の差分を選ぶ）---
            float zl = (x > 0)      ? depthBuf[o - 1]  : z;
            float zr = (x < RW - 1) ? depthBuf[o + 1]  : z;
            float zu = (y > 0)      ? depthBuf[o - RW] : z;
            float zd = (y < RH - 1) ? depthBuf[o + RW] : z;
            if (zl >= BIG) zl = z; if (zr >= BIG) zr = z;
            if (zu >= BIG) zu = z; if (zd >= BIG) zd = z;
            float ex, ey, ez, ax, ay, az, bx, by, bz;
            eye_pos(x, y, z, ex, ey, ez);
            float p1x, p1y, p1z, p2x, p2y, p2z;
            if (fabsf(zr - z) < fabsf(z - zl)) { eye_pos(x + 1, y, zr, p1x, p1y, p1z); ax = p1x - ex; ay = p1y - ey; az = p1z - ez; }
            else { eye_pos(x - 1, y, zl, p1x, p1y, p1z); ax = ex - p1x; ay = ey - p1y; az = ez - p1z; }
            if (fabsf(zd - z) < fabsf(z - zu)) { eye_pos(x, y + 1, zd, p2x, p2y, p2z); bx = p2x - ex; by = p2y - ey; bz = p2z - ez; }
            else { eye_pos(x, y - 1, zu, p2x, p2y, p2z); bx = ex - p2x; by = ey - p2y; bz = ez - p2z; }
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
                float t = clampf((z - camR * 0.5f) / (camR * 1.0f), 0, 1);
                int v = (int)((1 - t) * 255); small_[o] = rgb(v, v, v); continue;
            }
            if (p_mode == 3) {
                float t = clampf(thick_at(x, y) * 0.25f, 0, 1);
                small_[o] = rgb((int)(t * 90), (int)(t * 180), (int)(t * 255)); continue;
            }

            float vx, vy, vz; ray_of(x + 0.5f, y + 0.5f, vx, vy, vz);

            // --- 屈折。本家は背景をずらさない。平らな灰に透過率を掛けるだけ ---
            // 透過率 exp(−density·厚み·(1−diffuseColor))。diffuseColor=(0.085,0.6375,0.9) なので
            // 消散係数は (0.915,0.3625,0.1)。**青は赤の 1/9 しか吸われない**。
            // だから厚くなっても黒ではなく濃い群青で止まる。水の青は塗った色ではない。
            float d = p_absorb * thick_at(x, y);
            float tr = BG_GREY * fexp(-d * (1.0f - DIFF_R));
            float tg = BG_GREY * fexp(-d * (1.0f - DIFF_G));
            float tb = BG_GREY * fexp(-d * (1.0f - DIFF_B));

            // --- 反射（空）---
            float dot = vx * wx_ + vy * wy_ + vz * wz_;
            float rxv = vx - 2 * dot * wx_, ryv = vy - 2 * dot * wy_, rzv = vz - 2 * dot * wz_;
            float sr, sg, sb; sky_shade(rxv, ryv, rzv, sr, sg, sb);

            // --- フレネル（Schlick, F0 = 0.02）---
            float ct = clampf(-dot, 0, 1);
            float u = 1.0f - ct, u2 = u * u;
            float f = 0.02f + 0.98f * (u2 * u2 * u);

            float r = tr * (1 - f) + sr * f;
            float g = tg * (1 - f) + sg * f;
            float b = tb * (1 - f) + sb * f;

            // --- 鏡面。lightDir はワールドの (0,0,−1)、H = normalize(lightDir − rayDir)、pow(dot(H,n),250)
            float hx = 0.0f - vx, hy = 0.0f - vy, hz = -1.0f - vz;
            float hl = 1.0f / sqrtf(hx * hx + hy * hy + hz * hz + 1e-9f);
            float sp = clampf((hx * wx_ + hy * wy_ + hz * wz_) * hl, 0, 1);
            // 本家は pow(dot(H,n), 250)。2乗を8回で sp^256。
            // ★前の版はここを sp^64 にしていた（コメントは 256 と書いてあった）。
            //   64 だと highlight が広がって、水面が眠く見える。
            float s2 = sp * sp; s2 *= s2; s2 *= s2; s2 *= s2;     // sp^16
            s2 *= s2; s2 *= s2; s2 *= s2; s2 *= s2;               // sp^256
            float spec = s2;
            r += spec; g += spec; b += spec;

            // ★本家はトーンマップもガンマも掛けない。素通しでクランプするだけ。
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
        float t = clampf(s / 5.0f, 0, 1);
        px[(size_t)iy * FW + ix] = rgb((int)(60 + 195 * t), (int)(120 + 135 * t), 235);
    }
}

// 低解像度の流体を画面へ拡大（双一次）
static void upscale() {
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
    double b = nowms(); filter_depth(); filter_thick();
    double c = nowms(); shade(); upscale();
    tDepth += b - a; tFilt += c - b; tShade += nowms() - c;
}

// ================================================================ ABI
extern "C" {

KEEP void sim_reset();

KEEP int sim_w() { return FW; }
KEEP int sim_h() { return FH; }

static int p_scene = 0;
static void alloc_grid() { grid.assign((size_t)GX * GY * GZ * 4, 0.0f); }

KEEP void sim_reset() {
    realX = (float)GX; realY = (float)GY; realZ = (float)GZ;
    alloc_grid();
    if (p_scene == 1) setup_tank();
    else if (p_scene == 2) setup_splash();
    else setup_dambreak();
    setup_camera();
    bgDirty = true;
    frameNo = 0;
}

static void set_level(int lv) {
    if (lv < 0) lv = 0; if (lv > 4) lv = 4;
    p_level = lv;
    GX = LV_BOX[lv][0]; GY = LV_BOX[lv][1]; GZ = LV_BOX[lv][2];
    targetN = LV_NUM[lv];
    camR = LV_DIST[lv];
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
    small_.assign((size_t)FW * FH, 0);          // RS=1 でも足りるだけ取る
    depthBuf.assign((size_t)FW * FH, BIG);
    tmpBuf.assign((size_t)FW * FH, 0.0f);
    thickBuf.assign((size_t)TW * THH, 0.0f);
    thickTmp.assign((size_t)TW * THH, 0.0f);
    set_level(p_level);
    bgDirty = true;
    sim_reset();
    return 1;
}

KEEP void sim_set(int id, double v) {
    switch (id) {
    case 0: STIFFNESS = (float)v; break;
    case 1: GRAV = (float)v; break;
    case 2: SUBSTEP = (int)(v + 0.5); break;
    case 3: camAz = (float)v * 0.0174532925f; setup_camera(); break;
    case 4: camEl = (float)v * 0.0174532925f; setup_camera(); break;
    case 5: camR = (float)v; setup_camera(); break;
    case 6: p_radius = (float)v; break;
    case 7: p_absorb = (float)v; break;
    case 8: p_smooth = (float)v; break;
    case 9: p_mode = (int)(v + 0.5); break;
    case 10: { int lv = (int)(v + 0.5); if (lv != p_level) { set_level(lv); sim_reset(); } break; }
    case 11: DYN_VISC = (float)v; break;
    // 12: 箱の奥行き（本家の Box width スライダ）。狭めると水が押されて波が立つ。
    //     一気に動かすと粒子が壁を突き抜けるので、1フレームあたりの縮み幅を制限する。
    case 12: {
        float want = clampf((float)v, 0.5f, 1.0f) * GZ;
        float dv = want - realZ;
        if (dv < -0.35f) dv = -0.35f;
        realZ += dv;
        break;
    }
    case 13: p_hud = v > 0.5 ? 1 : 0; break;
    // 14: 流体の描画解像度の分母。1 = 全解像度（きれい・4倍重い）、2 = 半分（既定）
    case 14: set_rs((int)(v + 0.5)); break;
    default: break;
    }
}

KEEP void sim_action(int id) {
    if (id >= 0 && id <= 2) { p_scene = id; sim_reset(); }
}

KEEP double sim_get(int id) {
    switch (id) {
    case 0: return (double)pars.size();
    case 1: return (double)frameNo;
    case 2: return (double)p_level;
    case 3: return (double)realZ;
    case 4: return (double)GX;
    case 5: return (double)GY;
    case 6: return (double)GZ;
    case 7: return (double)camR;          // レベルを変えると C 側で引き直すので、JS から読み戻す
    default: return 0;
    }
}

// 画面をクリックしたら、そこの水面を叩く。深度バッファから当たった点を拾って、
// 半径 6 マスの中の粒子に手前向きのインパルスを足す。
KEEP void sim_click(double fx, double fy) {
    int x = (int)(fx / RS), y = (int)(fy / RS);
    if (x < 0 || y < 0 || x >= RW || y >= RH) return;
    float z = depthBuf[(size_t)y * RW + x];
    if (z >= BIG) return;
    float ex, ey, ez; eye_pos(x, y, z, ex, ey, ez);
    // 視空間 → ワールド
    float wy_ = cpit * ey - spit * ez, cz_ = spit * ey + cpit * ez;
    float wx_ = cyaw * ex + syaw * cz_, wz_ = -syaw * ex + cyaw * cz_;
    float px_ = camX + wx_, py_ = camY + wy_, pz_ = camZ + wz_;
    float dirx, diry, dirz; ray_of(x + 0.5f, y + 0.5f, dirx, diry, dirz);
    const float R = 6.0f, S = 9.0f;
    for (Par& q : pars) {
        float dx = q.x - px_, dy = q.y - py_, dz = q.z - pz_;
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > R * R) continue;
        float w = 1.0f - sqrtf(d2) / R;
        q.vx += S * w * dirx; q.vy += S * w * diry; q.vz += S * w * dirz;
    }
}

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
        char buf[192];
        Olivec_Canvas oc = olivec_canvas(px.data(), FW, FH, FW);
        snprintf(buf, sizeof buf, "UMI  MLS-MPM  particles %zu  box %dx%dx%d  substeps %d",
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
    if (const char* e = getenv("LEVEL")) { p_level = atoi(e); }
    if (const char* e = getenv("SCENE")) { p_scene = atoi(e); }
    sim_init(12345, 0);
    if (const char* e = getenv("SUB"))    sim_set(2, atof(e));
    if (const char* e = getenv("AZ"))     sim_set(3, atof(e));
    if (const char* e = getenv("EL"))     sim_set(4, atof(e));
    if (const char* e = getenv("DIST"))   sim_set(5, atof(e));
    if (const char* e = getenv("RADIUS")) sim_set(6, atof(e));
    if (const char* e = getenv("ABSORB")) sim_set(7, atof(e));
    if (const char* e = getenv("SMOOTH")) sim_set(8, atof(e));
    if (const char* e = getenv("MODE"))   sim_set(9, atof(e));
    if (const char* e = getenv("VISC"))   sim_set(11, atof(e));
    if (const char* e = getenv("RS"))     sim_set(14, atof(e));
    double squeeze = 1.0;
    if (const char* e = getenv("SQUEEZE")) squeeze = atof(e);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < frames; ++i) { if (squeeze < 1.0) sim_set(12, squeeze); sim_step(1); }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / frames;
    sim_render();
    std::vector<uint8_t> b((size_t)FW * FH * 3);
    for (int i = 0; i < FW * FH; ++i) {
        uint32_t c = px[i];
        b[(size_t)i * 3 + 0] = c & 0xFF; b[(size_t)i * 3 + 1] = (c >> 8) & 0xFF; b[(size_t)i * 3 + 2] = (c >> 16) & 0xFF;
    }
    stbi_write_png(out, FW, FH, 3, b.data(), FW * 3);
    double ds = 0, dmin = 1e9, dmax = -1e9, ymax = 0, vmax = 0;
    for (const Par& q : pars) {
        ds += q.dens; if (q.dens < dmin) dmin = q.dens; if (q.dens > dmax) dmax = q.dens;
        if (q.y > ymax) ymax = q.y;
        float v = sqrtf(q.vx * q.vx + q.vy * q.vy + q.vz * q.vz); if (v > vmax) vmax = v;
    }
    double tmax = 0, tsum = 0; int tn = 0;
    for (size_t i2 = 0; i2 < thickBuf.size(); ++i2) if (thickBuf[i2] > 1e-6f) { if (thickBuf[i2] > tmax) tmax = thickBuf[i2]; tsum += thickBuf[i2]; tn++; }
    printf("%s  particles=%zu  %.1f ms/frame (%d substeps)  box %dx%dx%d realZ=%.1f  dist=%.0f\n",
           out, pars.size(), ms, SUBSTEP, GX, GY, GZ, realZ, camR);
    printf("     density: mean=%.2f min=%.2f max=%.2f (rest %.1f) | vmax=%.2f ymax=%.1f | thickness max=%.2f mean=%.2f | filt<=%d px\n",
           ds / pars.size(), dmin, dmax, REST_DENS, vmax, ymax, tmax, tn ? tsum / tn : 0.0, filt_size_max);
    printf("     naiwake/frame:  MPM %.1f ms   depth %.1f ms   filter %.1f ms   shade %.1f ms   drawn %ld\n",
           tSim / frames, tDepth / frames, tFilt / frames, tShade / frames, dbgDrawn);
    return 0;
}
#endif
