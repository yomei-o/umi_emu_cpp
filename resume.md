# resume.md — 次に触るときの覚書

最終更新: 2026-08-17（2回目。1回目の版から物性・描画を全面的に書き直した）

## いまどうなっているか

3次元の水のシミュレータ。[webgpu-ocean](https://github.com/matsuoka-601/webgpu-ocean) を
**GPU を使わず C++ だけで**。MLS-MPM ＋ 画面空間の流体レンダリング。

**この版で「海」になった。** 前の版は水が塊に丸まって「球体」に見えていた。
原因は物理の選択（下の「一度目の間違い」）。本家のソースを定数まで読み直して合わせた。

- GitHub: https://github.com/yomei-o/umi_emu_cpp
- デモ: https://yomei-o.github.io/umi_emu_cpp/wasmdist/umi_os/
- 参考画像（合わせにいった写真）: `6b4cd02bf88e19e12f3d39e3456b558386f22796-thumb-1920x1080-64927.jpg`
- 本家のソースは読める場所に置いておくと早い: `git clone --depth 1 https://github.com/matsuoka-601/webgpu-ocean`

## すぐ動かす

```sh
cd /c/prog/claude/umi_emu_cpp

# WASM
EMSDK=/c/prog/emsdk/emsdk ./build.sh umi_os
cd wasmdist && python -m http.server 8765     # -> http://localhost:8765/umi_os/

# ネイティブ自己テスト（PNG＋毎段の時間。目視確認はこれが速い）
# ★このマシンに native clang は無い。emsdk の upstream clang が MSVC ターゲットで通る。
export MSYS2_ARG_CONV_EXCL='*' MSYS_NO_PATHCONV=1
/c/prog/emsdk/emsdk/upstream/bin/clang++.exe -O3 -std=c++17 \
  -Wno-deprecated-declarations -Wno-c99-designator -Wno-reorder-init-list \
  -I. -o umi_o3.exe src/umi_os.cpp
./umi_o3.exe 120 out.png
LEVEL=2 RS=1 ./umi_o3.exe 40 out.png          # 7万粒子・全解像度（絵を作るとき）
SCENE=2 ./umi_o3.exe 12 out.png               # 飛沫

# WASM の速度を測る（ブラウザを開かずに）
export EM_CONFIG=/c/prog/emsdk/emsdk/.emscripten
export PATH="/c/prog/emsdk/emsdk/upstream/emscripten:$PATH"
em++ -O2 -std=c++17 -msimd128 -I. src/umi_os.cpp -sMODULARIZE=1 -sEXPORT_NAME=createSim \
  -sENVIRONMENT=node -sALLOW_MEMORY_GROWTH=1 -sEXPORTED_RUNTIME_METHODS=cwrap,HEAPU8 \
  -sEXPORTED_FUNCTIONS=_sim_init,_sim_w,_sim_h,_sim_reset,_sim_step,_sim_render,_sim_click,_sim_set,_sim_action,_sim_get \
  -o /tmp/sim.js
/c/prog/emsdk/emsdk/node/22.16.0_64bit/bin/node.exe bench.js   # cwrap で叩くだけ
```

環境変数: `LEVEL SCENE SUB AZ EL DIST RADIUS ABSORB SMOOTH MODE VISC RS SQUEEZE`

## ファイル

| ファイル | 中身 |
|---|---|
| `src/umi_os.cpp` | 物理・描画すべて（これ1本、約900行） |
| `build.sh` | em++ ラッパ（`-msimd128` 付き） |
| `wasmdist/umi_os/index.html` | JSハーネス（hanabi / taki と同じ ABI） |
| `olive.c` / `stb_image_write.h` | HUD文字 / ネイティブPNG出力 |
| `shot_*.png` | README 用のスクリーンショット（RS=1 で焼いたもの） |

## ABI（sim_set の id）

★前の版から**意味が変わっている**。単位は全部「格子1マス」。

| id | 意味 | 既定 |
|---|---|---|
| 0 | 剛性（Tait の k） | 3.0 |
| 1 | 重力 | 0.3 |
| 2 | サブステップ | 2 |
| 3/4/5 | 視点 方位[°] / 仰角[°] / 距離[マス] | 135 / 15 / lv依存 |
| 6 | 球の半径[マス] | 0.6 |
| 7 | density（吸収のスケール） | 1.5 |
| 8 | 表面の平滑化（窓幅の倍率） | 1.0 |
| 9 | 表示モード 0水 1法線 2深度 3厚み 4粒子 | 0 |
| 10 | 粒子数プリセット 0..4 | 1 |
| 11 | 動粘性 | 0.1 |
| 12 | 奥の壁の位置（GZ に対する比 0.5..1.0）**毎フレーム呼ぶ** | 1.0 |
| 13 | HUD 0/1 | 1 |
| 14 | 描画解像度の分母（1=全解像度, 2=既定） | 2 |

`sim_get`: 0=粒子数 1=フレーム 2=レベル 3=realZ 4/5/6=GX/GY/GZ **7=カメラ距離**
`sim_action`: 0=ダムブレイク 1=水槽 2=飛沫。`sim_click(x,y)` は水面を叩く。

## 粒子数プリセット（本家 main.ts の4段 ＋ CPU用に軽い段）

| lv | 箱 | 粒子数 | 距離 | ネイティブ | WASM |
|---|---|---|---|---|---|
| 0 | 30×22×45 | 20,000 | 52 | 24 ms | 30 ms |
| **1** | **35×25×55** | **40,000** | **60** | **35 ms** | **44 ms** |
| 2 | 40×30×60 | 70,000 | 70 | 50 ms | 62 ms |
| 3 | 45×40×80 | 120,000 | 90 | 72 ms | 83 ms |
| 4 | 50×50×80 | 200,000 | 100 | 139 ms | 131 ms |

lv1〜4 は本家の `mlsmpmInitBoxSizes / mlsmpmNumParticleParams / mlsmpmInitDistances` そのまま。

---

# 一度目の間違い（★ここが一番大事。同じ道を二度行かないこと）

「本家と同じものを書いているつもり」で**別の流体を書いていた**。
教科書（taichi の mpm88 系）の MLS-MPM をそのまま3次元にしたのが間違いだった。

| | 一度目 | 本家 = 今 | 症状 |
|---|---|---|---|
| 単位 | m・s（dx=5cm, g=9.8, E=400） | **格子1マス=1, g=0.3, dt=0.20** | ― |
| 圧力 | 体積比 J から `E·(J−1)` | **格子から測った密度の Tait**<br>`max(0, 3·((ρ/4)⁵−1))` | **J 方式は負圧＝引力になる。水が丸まって「球体」** |
| c/v | c=20, v≈4.4 → マッハ0.2 | c≈1.9, v≈4 → **マッハ2** | 硬すぎて dt が取れず、1フレームでほぼ動かない |
| 粘性 | なし | `stress += 0.1·(C+Cᵀ)` | 表面が毛羽立つ |
| 箱 | 40×20×26（浅い平鍋） | **40×30×60（奥に長い）** | 海面の広がりが出ない |
| カメラ | 方位20°、距離＝幅の1.15倍 | **方位45°、距離＝幅の1.75倍** | 画面いっぱいの塊に見える |
| 厚み | 手前の粒子だけ | **全粒子** | 厚い所が薄く出る |

**気づき方**: 本家は `p2g_2.wgsl` で毎ステップ密度を測り直している（nialltl の記事いわく
「体積の推定が甘いと dt を小さくせざるを得ない。毎ステップ測り直せば dt を大きく取れる」）。
J を持ち回す教科書版とは**別の手法**。ここを読み飛ばしていた。

## 数字の出どころ（自分で引き直せるように）

- 音速 `c² = dp/dρ = 3·5·ρ⁴/4⁵`。ρ=4 で c=1.94。dt=0.2 なら `c·dt=0.39` マス < 1 で CFL は足りる
- 落下速度 `v=√(2·0.3·24)=3.8`。**c より速い**。本家の水はそれくらい柔らかい
- 1フレームで進む時間は `2·0.2=0.4`。箱の高さ30を落ちるのに `√(2·30/0.3)=14.1` → **35フレーム**。
  一度目は 259 フレームかかっていた（7.4倍のスローモーション）
- 厚みの単位。`0.05 · √(1−r²)` を全粒子ぶん足すと、光路 L マスあたり
  `0.05 · n · πr² · (2/3) · L = 0.137·L`（n=3.64, r=0.6）。L=20 で 2.7。
  **粒子密度や半径を変えたら `TALPHA` を引き直さないと色が合わない**（`0.05·(3.64/n)·(0.6/r)²`）
- 双方向フィルタの窓 `filter_size = 1.2 × 画面上の球の半径[px]`
  （本家 `projected_particle_constant = 12·直径·0.05·(画面高/2)/tan(fov/2)` を depth で割った形）

---

# 課題（ここから続き）

## 1. 泡・飛沫が本家より少ない

参考写真の中央には**白い飛沫の雲**が立っている。こちらは水面の白い縁までは出るが雲にならない。
本家 README も TODO に *"Unified Spray, Foam and Bubbles for Particle-Based Fluids"* を挙げている。
やるなら：速度と表面性から泡粒子を吐いて、白く（吸収させず）別に描く。

## 2. 速度：MPM が支配的になった

lv1 で MPM 12.6 / depth 6.7 / filter 8.3 / shade 7.0 ms。描画側はだいぶ削れたので、次は MPM。

- P2G を **1パスにできないか**。いまは P2G1（質量）→ P2G2（密度を測って応力）で27節点を2回舐めている。
  P2G2 の密度ループ（27節点）だけでも消せれば 1/4 は減る。
  ただし密度は「格子の質量を粒子位置で補間した値」なので、格子側で先に作るのは同値にならない。
- 重み `wx/wy/wz` は3パスで作り直している（bspline×3）。粒子ごとに9floatを持てば省ける（効果は小）
- **スレッド**。WASM の pthreads は GitHub Pages では使えない（COOP/COEP ヘッダが送れず
  SharedArrayBuffer が無効）。自前ホストか Cloudflare Pages なら 4 コアで 3〜4倍。判断が要る

## 3. まだやっていないこと

- **壁の可視化がない**。水が宙に浮いて見える。うっすら箱の輪郭か床の影があると据わりが良い
- マウスで視点を回せない（スライダのみ）。`sim_click` は水を叩くのに使ってしまった
- 本家の cubemap（写真）を焼き込む案。いまは手続きの夕空。地平の白い霞は出ているが雲は無い
- 全解像度（`解像度→高精細`）は 4 倍重い。深度だけ全解像度で撒いて平滑化と着色は半分、という
  分離を試す余地はまだある

---

# 踏んだ罠

## 物理

- **教科書の MLS-MPM は本家と別物**（上に書いた）。負圧を捨てる `max(0,…)` の1行が効く
- 間隔 0.65 の格子は密度 3.64（rest 4）で**少し足りない**。置いただけだと自重で沈んで壁を叩く。
  静かな水面から始めたい場面は `settle()`（粘性を上げて70サブステップ回し、速度を捨てる）を通す
- 本家の初期ジッタは **3成分に同じ値**（`jitter = 2.0*random()` を x,y,z に足す）。対角にずれる。
  変な実装だが、そのまま真似ないと粒の並びが変わって初期の崩れ方が違う
- 壁は**格子側（速度成分を0）と粒子側（予測位置のバネ k=3, stiffness=0.3）の二重**。
  格子だけだと角に溜まる

## 描画

- **消散係数は (0.915, 0.3625, 0.1)。青は赤の 1/9 しか吸われない。** 水の青は塗った色ではない
- **屈折で背景をずらさない。** 平らな灰 0.8 に透過率を掛けるだけ
- **トーンマップもガンマも掛けない。** 素通しでクランプするだけ。
  掛けると背景（灰0.8→ガンマで211）と水の中の背景色（204）が違ってシルエットに枠が出る
- **`pow(dot(H,n), 250)` を2乗の繰り返しで書くときは回数を数える。**
  `s2=sp²; s2*=s2; s2*=s2; s2*=s2; spec=s2*s2*s2*s2` は sp^64（sp^256 ではない）。
  コメントに 256 と書いてあったのに 64 だった。highlight が広がって水面が眠く見える
- **本家の cubemap は青空ではなく夕暮れ**（"Industrial Sunset 02 (Pure Sky)"）。
  水面を走る暖色の筋はその映り込み。青空だと出ない
- **深度の平滑化は4往復**。1往復だと粒がそのまま残る（本家 `fluidRender.ts` の `for (iter<4)`）
- **厚みのぼかしは σ≈10px（本家の画面高756で30px窓）。** かなり大きい。
  小さいと粒ごとの色むらが残って「球の集まり」に見える
- **「内部の粒子を描かない」間引きは危ない。** 質量格子の6近傍で判定すると表面の粒子まで落ちて
  水面に穴が空く。代わりに「球のいちばん手前でも既存の深度より奥なら捨てる」という
  **厳密な**早期打ち切りを使う。手前から撒くとよく効く（粒子はセル順に並んでいるので、
  カメラが +z 側なら配列を逆から舐めるだけ）
- 矩形だけ処理すると継ぎ目が出る。拡大が双一次で1画素はみ出して読むので、着色は矩形より3画素広く

## 実装

- **推測せずに毎段の時間を測ること。** `tSim/tDepth/tFilt/tShade` を出す仕組みが入っている
- **SIMD は格子の持ち方で決まる。** (vx,vy,vz,m) の4つ組にしてあるので 1節点 = 1ベクタ。
  27節点のループが load/fma/store 各1回になって MPM が 1.8倍。
  `std::vector<float>` は16バイト境界を保証しないので `aligned(4)` のベクタ型で受ける
- **SIMD 化すると同じ乱数でも100フレーム先の絵が変わる**（加算順が変わる → 丸めが変わる → カオス）。
  正しさの確認は**フレーム1〜20の統計が一致するか**で見る。ここが合っていれば式は同じ
- このマシンに native clang は無い。**emsdk の upstream clang** が MSVC ターゲットで通る
  （`-Wno-c99-designator -Wno-reorder-init-list` を足すと olive.c の警告が黙る）
- ブラウザを開かずに WASM の速度を測るなら `-sENVIRONMENT=node` で焼いて emsdk 同梱の node で叩く
- **`MSYS_NO_PATHCONV=1` / `MSYS2_ARG_CONV_EXCL='*'` を立てたシェルで `./build.sh` を呼ぶと壊れる。**
  `.emscripten` が `NODE_JS = dirname(EM_CONFIG) + ...` を組むので、EM_CONFIG が
  POSIX パスのまま Windows の python に渡って `NODE_JS not set in config` になる。
  この2つは native clang を叩くときだけ立てて、em++ の呼び出しとは**別のシェルにする**
- Windows の bash ヒアドキュメントで C 文字列の `\n` が**本物の改行に潰れる**。
  `\n` を含むパッチは Edit ツールを使うこと
