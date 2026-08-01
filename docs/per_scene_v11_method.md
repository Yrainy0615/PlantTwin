# Per-Scene Physics Optimization (v11)

输入：GaussianPlant 静态重建 + 单视角 dynamic video（手拉/风吹）+ COLMAP 相机
输出：可重驱动的 sim-ready 数字植物（cleaned tree + 每类物理参数 + 接触力轨迹）

---

## 1. 输入

| 来源 | 内容 |
|---|---|
| GaussianPlant `source/` | `point_cloud.ply` (ApP — N≈708k for newplant9), `branch.ply`, leaf clusters `cluster_*.ply` (43 leaves) |
| GaussianPlant `output/` | `branch_mst.ply` (noisy branch graph), `branch_tube.ply` |
| COLMAP `sparse/0/` | `cameras.bin`, `images.bin`（用 `IMG_1388.JPG` 的 pose） |
| Video | `motion.mp4` (hand-pull, 由 Wan2.2-I2V 生成) |
| Contact bundle | `contact.pt` — GroundedSAM hand mask + CoTracker3 + PhysTwin filter 得到 per-frame anchor node id、pin pixel、stem tracks |

## 2. Stage 0 静态结构处理 (offline, deterministic)

**0.A Branch graph cleanup** (`models/structure/graph_cleanup.py`)
- Root：`argmin(nodes[:, 1])`（+Y 在 3DGS world 是 up，根 = 最低节点）
- BFS rooting → parent / depth / subtree_size
- Per-edge radius：tube mesh 的 vertices 到每条 edge centerline 的平均距离
- Stem vs branch：从 root 沿最大子树 child 走的路径为 stem

**0.B Leaf attachment** (`models/structure/leaf_attachment.py`)
- 每片 leaf cluster：PCA 拟 disk (centroid c_k, normal n_k, radius r_k)
- Leaf-root candidate：cluster 内离任何 branch node 最近的点
- 评分：score = surface_distance − terminality_bonus * is_terminal_child
- 取 argmin edge，snap 到 cylinder surface 得 (parent_edge, u, surface_point, rest_length, rest_direction)
- 43 leaves → 34 unique parent edges (newplant9)

**LBS skinning** (`models/structure/skinning.py`)
- 每个 ApP 二选一：绑到最近 bone (signed surface distance) 或 leaf disk
- `max_dist=0.3` 阈值之外的 ApP 标记为 static（背景/盆栽不跟动）
- newplant9 上：407k/708k → static（背景）, 剩下 300k 跟着结构动
- Local 坐标在 canonical pose 计算一次，保证 rest 渲染像素级 match

## 3. 动力学模型

**3.1 Articulated chain on branch tree** (`simulation/articulated_chain.py`)
- 每个非 root 节点 i：state = (θ_i ∈ ℝ³ axis-angle joint angle, ω_i 角速度)
- Forward kinematics: `R_i = R_parent @ exp_so3(θ_i)`，`pos_i = pos_parent + R_i @ (rest_i − rest_parent)`
- Joint dynamics (semi-implicit Euler):
  ```
  τ_i = −k_i θ_i − c_i ω_i + τ_ext_i
  ω_i += dt · τ_i / I_i
  θ_i += dt · ω_i
  ```
- External force `f_j` 在 node j 上 → 对所有 ancestor i 的 torque：`τ_ext_i += (pos_j − pos_i) × f_j`
- Per-type 参数（仅 6 个 scalar）：`k_stem, k_branch, c_stem, c_branch, inertia`，type 由 stem-path 标注决定

**3.2 Petiole joint per leaf** (在 `PerScenePhysicsModel._rollout_petiole`)
- 每片叶子 base-excited damped oscillator，driven by parent bone 的 angular acceleration α_parent
- `omega' = −k_petiole · θ − c_petiole · ω − α_parent`（I 归一化为 1）
- α_parent 由相邻两帧 child bone rotation matrix 的 skew-symm vee 算出（小角度精确）
- Leaf disk rotation：`R_leaf = R_child @ exp_so3(θ_petiole)`
- Leaf disk center：`surface + R_leaf @ (rest_dir · rest_length)`
- 全局两个 scalar：`k_petiole, c_petiole`

**3.3 Leaf ApP 绑定（v11 取消 intra-leaf spring-mass）**
- Leaf-bound ApP 用 disk-frame rigid LBS：`ap_world = leaf_center + R_leaf @ local_leaf`
- 整片叶子的运动完全来自 petiole joint（已优化）+ parent bone（已优化）
- 取消原因：在 monocular RGB+stem_proj 监督下，intra-leaf spring-mass 的 (k_leaf, c_leaf, m_leaf) 梯度被淹没在数值噪声里（实测训练全程不动），同时引入 IDW interpolation 在稀疏 cluster 采样上的"细碎" artifact

**3.4 Contact force trajectory** (`simulation/contact_force.py`)
- Per-frame 3D force 作用在唯一 anchor node（从 contact bundle 投票得到 most-frequent node）
- 正则：temporal smoothness + sparsity（防止 force 过激）

## 4. 损失

| Term | 权重 | 内容 |
|---|---|---|
| `L_RGB` | 1.0 | 渲染 vs target video 像素 L2 |
| `L_stem_proj` | 0.005 | CoTracker stem tracks 的 2D 投影 L2（在 visible 子集上） |
| `L_pin` | 0.001 | Anchor node 2D 投影 vs detected hand pin per-frame |
| `L_contact_smooth` | 0.001 | force[t+1] − force[t] 的 L2 |
| `L_contact_sparse` | 0.001 | force L1 |

Renderer：`models/renderer/gaussian_renderer.py`（GaussianRenderer，sh_degree=0 用 RGB 直接）
COLMAP camera：raw R, t（不再做 diag(1,−1,1) 翻转；之前 projection 函数里多一个 Y 反号被去掉）

## 5. 优化设置

| | 值 |
|---|---|
| Frames | 24 |
| Start frame | 12（跳过 hand 还没接触植物的前 12 帧） |
| dt | 0.02 |
| Renderer res | 256 × 256 |
| Optimizer | Adam |
| lr | 1e-2 |
| Steps | 80 |
| Anchor | bundle 投票（newplant9: node 118） |

## 6. v11 收敛 (newplant9 / IMG_1388 hand-pull)

```
[000] loss=7.60e-01 rgb=3.6e-02 stem_proj=90.2 pin=273  k_pet=79  c_pet=2.97
[040] loss=6.21e-01 rgb=4.0e-02 stem_proj=71.5 pin=223  k_pet=90  c_pet=2.88
[079] loss=6.15e-01 rgb=4.1e-02 stem_proj=71.2 pin=218  k_pet=132 c_pet=4.15
```
- Total loss 0.76 → 0.61（收敛）
- stem_proj 90 → 71（branch 投影对齐）
- pin 273 → 218（接触位置对齐）
- k_pet 79 → 132（叶柄更刚）
- k_branch 7.9 → 3.0（branch 更软，配合更大的 force 让 stem 投影动起来）

## 7. 已知 limitations

1. **Intra-leaf bending 不可识别** — 单视角 RGB 监督拿不到 leaf 内部弯曲的梯度信号。要识别需要 (a) leaf-tip CoTracker 轨迹做 tip-projection loss，或 (b) 多视角，或 (c) flow loss。
2. **Contact force depth ambiguity** — 单视角 hand pose 深度模糊，force 在 z 方向不可识别，靠 contact-pin 软约束 + 时间平滑兜底。
3. **力作用点 = 单 node** — 实际 hand grip 是 moment + distributed force，v0 简化为 point force，systematic 投影误差可能没被解决。
4. **Branch tree quality 决定一切** — Stage 0 的 root 选择、leaf attachment 是 deterministic 的，错了后面学不回来。

## 8. 下一步候选

- (A) 加 `L_leaf_tip`：用 CoTracker3 跟 leaf 端点，给 leaf 一个有效监督；可同时给 intra-leaf spring-mass 解锁。
- (B) 加 SAM 分割 mask 的 silhouette IoU loss，对 leaf 边界做 weak 监督。
- (C) 把 contact 改成 anchor 附近 small region 的 distributed force，缓解 single-point 假设。
- (D) Wind-only 数据上验证 identifiability：把 IMG_6698_wind_left 等 4 段 wind 视频跑通，验证学到的 stem/branch stiffness ratio 满足 `k_stem > k_branch > k_petiole` 的植物学先验。
