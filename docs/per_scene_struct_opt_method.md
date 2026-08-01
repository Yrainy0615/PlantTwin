# Motion-Cue Structure Optimization (Structure-in-the-Loop)

在 v11（物理参数 + 接触力，结构 frozen）的基础上，把 **branch tree 的结构本身**也变成可优化对象：
用 video motion fit 的残差作为信号，找出"现有 topology 表达不了的运动"，并据此
(1) 微调现有节点的 rest 位置，(2) 对表达力不足的 branch 做 densification。

核心 idea：**结构误差会伪装成运动误差**。如果 branch tree 在某处少了一个关节 / 节点位置偏了，
optimizer 没法靠物理参数（k, c）把那段运动 fit 上去——这份 fit 不掉的残差，正是
"这里结构不够"的信号。把它反传到一个 per-node 的 rest 位置修正量 δ 上，δ 大的地方就是结构该被
加密 / 修正的地方。

---

## 1. 可学习量：per-node rest 位置修正 δ

给 branch tree 的每个节点 i 一个可学习的 3D 修正向量 δ_i：

```
rest_pos_eff[i] = nodes[i] + δ_i           # δ ∈ ℝ^{N×3}, nn.Parameter
```

`rest_pos_eff` 取代原始 `tree.nodes` 喂给 forward kinematics：

```
pos_i = pos_parent + R_i · (rest_pos_eff[i] − rest_pos_eff[parent])
```

直觉：δ 改变的是**骨骼的静止几何**（bone 的长度 / 朝向 / 关节位置），而不是某一帧的姿态。
所以一个 δ 会同时影响所有帧的 FK，是一个"结构级"的自由度，区别于逐帧的 θ（姿态级）。

### 1.1 关键约束：canonical pose 必须像素级不变

δ 改了 rest 几何，但 t=0 的渲染必须和原始 GaussianPlant 重建**逐像素一致**——否则
optimizer 会用 δ 去 hack 第一帧的 RGB，而不是去解释运动。做法是在 LBS 里把 δ 的影响
**在 local 坐标上抵消掉**：

```
# bone-bound ApP：local_eff = local − δ[parent]
ap_world = pos_parent + R_parent · (local_bone − δ[parent_idx])

# leaf-bound ApP：同理用 δ[leaf_attach_parent] 抵消
```

效果：canonical pose（所有 R = I）下 `ap_world` 与原始完全相等，δ 只在**有旋转的运动帧**
才显现作用。这保证了 δ 的梯度纯粹来自 motion fit，不来自 t=0 的外观。

---

## 2. 结构正则

δ 是 N×3 的高维量，单视角监督下严重欠定。两个正则把它压成"稀疏 + 平滑"：

```
L_struct_l2     = mean_i ‖δ_i‖²                                  # 越小越好：尽量不动
L_struct_smooth = mean_{(p,c)∈E} ‖δ_c − δ_p‖²                    # 沿 edge 平滑：相邻节点修正一致
```

- **L2（w=0.5）**：让 δ 默认贴近 0，只有 motion 残差真的需要时才让某个 δ 长大 → 自动稀疏化，
  δ 集中在"结构真不对"的少数节点。
- **Smooth（w=2.0）**：沿树边惩罚 δ 的跳变。防止单个节点孤立地乱跳，让修正以"一段 branch
  整体平移/旋转"的物理合理方式出现。smooth 权重比 L2 大，因为我们更怕高频噪声解。

总损失：

```
L = L_RGB + 0.005·L_stem_proj + 0.001·L_pin
      + 0.001·(L_contact_smooth + L_contact_sparse)
      + 0.5·L_struct_l2 + 2.0·L_struct_smooth
```

δ 用**更小的 learning rate**（`lr · struct_lr_scale`, scale=0.2）和物理参数分组优化——
结构是慢变量，物理参数是快变量，分开步长避免结构抖动。

---

## 3. Densification：用 δ 决定在哪里加节点

δ 收敛后，‖δ_c‖ 大的 edge = "motion fit 想在这里要更多 articulation，但原 topology 给不了"。
对这些 edge 做中点分裂，给 branch 增加一个真正的关节自由度：

### 3.1 选边（`pick_split_edges`）

每条有向边 (p, c) 用其 child 的 δ 范数打分，取 top-K：

```
score(p, c) = ‖δ_c‖
筛选：edge_length ≥ 0.3·median_edge_length   （不分裂已经很短的边）
      score ≥ min_delta_norm                  （太小的残差不值得加节点）
```

### 3.2 分裂（`split_edges`）

每条选中的 (p, c) 插入中点 M，一条边变两条：

```
M = ½(nodes[p] + nodes[c])                    # rest 中点
(p, c)  →  (p, M) + (M, c)
parent[M] = p,  parent[c] = M                 # c 的整个子树深度 +1
edge_type / edge_radius 继承自原边
edge_length 各取原边的一半
```

新增的 M 节点带来一个**新的关节**——FK 在 M 处可以再转一次，于是原本"一根直 bone 拉不出来"
的弯曲，现在有了表达自由度。densify 后重新跑 §1–2 的优化（warm-start 物理参数 + 把旧 δ
zero-pad 到新节点数），让新关节的 δ 和物理参数继续 fit。

整个 pipeline 是一个 **outer densify loop**：
```
v11 (物理 only)
  └─► v12: + δ (learn-struct)            ← 找出结构残差
        └─► v13: densify top-K δ-edges   ← 在残差大处加节点
              └─► 再优化 δ + 物理          ← 收敛
```

---

## 4. 实测结果 (newplant9 / IMG_1388 hand-pull, 60 steps each)

| 版本 | N | loss | RGB | stem_proj | pin | max‖δ‖ |
|---|---|---|---|---|---|---|
| v11 (物理 only) | 424 | 0.615 | 0.041 | 71.2 | 218 | — |
| v12 (+δ) | 424 | ~0.20 | ~0.042 | ~27 | ~12 | 35 cm |
| v13 (densify 12 edges) | 436 | ~0.135 | 0.042 | ~14 | ~5.6 | ~54 cm |

**关键观察**：RGB 几乎不动（~0.042），但 stem_proj / pin 大幅下降。
说明 δ 改善的是 **branch 投影对齐（运动表达力）**，不是逐像素外观——
这正符合 δ 的设计（canonical 外观被锁死，δ 只influence运动）。

**这也暴露 limitation**：RGB 这一侧几乎没有结构梯度。要让 δ 同时改善像素级 fit，需要
leaf-tip CoTracker 轨迹或 silhouette IoU 这类**能把 motion 误差喂到 RGB/mask 上**的监督，
否则结构优化只能被 stem_proj / pin 这类稀疏 2D 点监督拉动。

---

## 5. 与 dense KNN spring-mass 类方法的区别

- Spring-Gaus / OmniPhysGS：在 dense 点云上学 KNN 弹簧，结构无植物学含义、参数不可识别。
- 本方法：结构始终是**离散 branch tree**，δ 只是对已有可解释节点的有界修正，densify 也只在
  树上插中点——拓扑始终保持"根→茎→枝→叶柄→叶"的语义。小、可解释、sim-ready。
- 信号来源不同：不是让结构去拟合点云，而是让 **video motion 的 fit 残差**反过来告诉结构
  "哪里缺自由度"。结构服务于运动表达，而非外观重建。

## 6. Shape + Topology 约束（防止 node 飘出 branch）

### 问题
δ 是欠约束的自由量。densify 插入的中点节点、以及被 motion 残差推动的原节点，会**飘出
branch 的几何范围**（实测 warm-start 状态下有 18/436 个节点在 branch 之外），并可能引入
局部"kink"——某个节点的运动方向和相邻节点相反，物理上不合理。L2/smooth 正则只约束 δ 的
大小和沿边平滑，并不知道"branch 实际长在哪里"。

### Idea
用 GaussianPlant 的 **dense tube-surface 点云**（newplant9: 51k 点）作为几何先验——
skeleton 节点本身给不了 branch 的体积信息。两个约束：

**6.1 几何包含约束 `L_geom`（shape）**
每个 effective rest 节点 `p_i = nodes_i + δ_i` 到 dense branch 点云的最近邻距离 `d_nn(i)`，
只惩罚超出容差带 `r_tol` 的部分（单边 hinge）：

```
d_nn(i)  = min_v ‖p_i − branch_point_v‖
L_geom   = mean_i ( max(0, d_nn(i) − r_tol) )²
```

- `r_tol` 自适应：取**原始节点**到点云最近邻距离的 95 分位 × scale(=2)，newplant9 上 ≈ 0.085。
  原节点本就在 branch 内，所以这个带宽就是"什么叫在 branch 里"的数据驱动定义。
- **单边 hinge** 的关键性质：节点可以**沿 branch 自由滑动**（带内 penalty=0），只有**离开
  branch 体积**才被拉回。这区别于"L2 拉回原位"——后者会阻止合理的沿枝调整。
- 对 thin branch（半径 ~0.016）整个 tube 内部都在 r_tol 内，所以"近 surface"≈"在 branch 内"。
- 点云 subsample 到 8k，每步 `cdist(N, 8k)` 很快；分块算最近邻控制显存。

**6.2 运动方向一致性约束 `L_motion`（topology）**
相邻节点（树上的 parent p、child c）的逐帧位移方向应一致。位移
`m_i(t) = pos_i(t) − p_i`，用 magnitude-gated cosine：

```
L_motion = Σ_{t,(p,c)} w · (1 − cos(m_p(t), m_c(t)))  /  Σ_{t,(p,c)} w,
其中 w = ‖m_p(t)‖ · ‖m_c(t)‖   （只在两端都动时才计方向，避免静止帧噪声主导）
```

- 直接惩罚"相邻节点运动方向相反"的 kink，正是 densify 新节点最容易引入的非物理模式。
- 用 cosine 而非位移差 `‖m_c−m_p‖²`：后者会惩罚 tip 比 base 动得多这种**合理**现象；
  cosine 只管方向不管幅度。

### 总损失（加上约束后）
```
L = L_RGB + 0.005·L_stem_proj + 0.001·L_pin
      + 0.5·L_struct_l2 + 2.0·L_struct_smooth
      + 20·L_geom + 0.5·L_motion
      + 0.001·(contact smooth + sparse)
```

### 实现 / 测试
- `optimization/struct_constraints.py`：`geometric_containment_loss`、
  `motion_direction_consistency_loss`、`estimate_geom_margin`、`subsample_points`
  （纯 tensor op，可单独单元测试）。
- `tests/test_struct_constraints.py`：7 个单元测试——geom 带内为 0 / 带外正且梯度把
  逃逸节点拉回 / 沿枝滑动免罚；motion 同向为 0 / 反向为正且梯度对齐 / 小幅度被 gate。
- `scripts/pipeline/optimize_per_scene.py`：新增 `--w-geom`(默认 20) / `--w-motion`(默认 0.5) /
  `--geom-margin-scale`(默认 2) / `--geom-max-points`(默认 8000)，每步记录
  `geom / motion / geom_n_out` 进 history。

## 7. 已知 limitations / 下一步

1. RGB 侧无结构梯度（见 §4）→ 加 leaf-tip track / silhouette loss。
2. δ 在 z（深度）方向欠定（单视角）→ 多视角或 depth prior。
3. densify 目前只插中点、只加密不删枝 → 可加 edge collapse（δ≈0 且冗余的边合并）做双向。
4. top-K 是固定 K → 可改成基于 δ 分布的自适应阈值。
5. `L_geom` 用 surface 点云对 thin branch 是"近 surface≈在内部"的近似；粗 branch（如根部
   半径 0.6）会被允许的带宽偏小，可改成 per-node `r_tol = edge_radius + margin`。

---

## 8. Motion-in-the-Loop：用运动直接改善 structure（深度 + 拓扑）

§1–7 是在**冻结骨架**上学 δ，motion 只通过残差间接进入，改不了 StPr 本身、也不考虑
appearance 与 skeleton 的几何 binding（正是 §7 limitation #2「单视角下 δ 深度欠定」的根源）。
本节把 motion 作为**直接监督**接进结构优化：AppGas 经几何 binding 随 skeleton 运动，单目视频
的运动残差反传去**修正节点深度**，运动 rigidity 反传去**识别 branch 连接**。

### 8.0 数据现实：为什么用合成 GT
`data/*/motion.mp4` 全是 **diffusion 生成**的（meta.json 有 prompt/seed/guidance），叶片
布局和 reconstruction 对不上、还重新取景+幻觉遮挡 → **无法逐像素监督**。所以做受控验证时，
用 reconstruction 自己渲染忠实运动视频（完美对齐 = GT motion），把真结构打乱再恢复，从而能
**定量**判断 motion 是否改善结构。（在真实结构上落地需要这株植物的**忠实拍摄**视频。）

### 8.1 几何 binding（`models/structure/edge_binding.py`）
关键前提：节点位置要能在渲染里被"看见"，才能被 RGB 约束。
- **branch AppGas → 边局部坐标系**：每个点参数化为沿轴坐标 `t` + 边局部系垂直偏移
  `(ou, p1, p2)`，世界坐标 `= P[a] + t·d + ou·u + p1·e1 + p2·e2`（`d=P[b]−P[a]`，
  `e1,e2` 由 `u` 确定性构造）。移动/旋转边 → AppGas 真实跟着动（位置+旋转都敏感）。
- **leaf AppGas → leaf instance 刚体**：按最近 cluster 质心归到 instance，instance attach 到
  最近 branch 节点，随该节点 FK 帧**整体刚体旋转**摆动。这样海量叶片不再是死重——
  **叶片大幅摆动反过来约束它所挂的 branch 节点**（用 leaf-branch attachment 做杠杆）。
- rest 位姿精确还原（parity 1.9e-6）；canonical 多视图渲染对齐 gsplant（33dB）。

### 8.2 信号一：深度（video RGB）
单目静态只能定 image-plane，深度在零空间里乱漂。摆动视频里同一节点的运动弧**随深度不同
投影不同** → 提供深度线索。损失 = 多视图静态 RGB（锚定 in-plane）+ 单目运动视频逐帧 RGB
（修深度），AppGas 由 `pos_t = FK(θ_t, P+δ)` + 上述 binding 重建。

### 8.3 信号二：拓扑（3D rigidity）
静态几何对**连接**几乎无知：两条不同枝上的点可以空间很近（短的假候选边）却不相连。运动揭示它：
**真实枝段在弯曲中长度恒定（刚性），跨枝假边两端独立运动会被拉伸**。对候选边集
（真树边 ∪ kNN）每条边给可学权重 `w_e = σ(l_e)`，用应变把 w 拉向低应变（真）边：

```
strain_e = std_t‖pos_a(t) − pos_b(t)‖ / len_e        （运动中相对长度变化）
L_rigid  = mean_e ( w_e · strain_e / mean(strain) )   ← w 越大要求越刚
```

实测：真边 strain 中位数 ~1e-6，假边 ~1e-2.5（差 4 个数量级）。
**注意**：rigidity 信号在 **3D** 里极强、但**原始 2D 投影测不出来**（前缩让真边 2D 长度也变，
AUC 仅 0.52）——必须靠联合优化把视频运动**抬升到 3D** 才能用上，这本身就论证了
structure-in-the-loop 3D 联合优化的必要性。

### 8.4 联合优化器（`scripts/pipeline/fuse_motion_structure.py`）
同一个循环里**同时**优化 δ（节点位置）和 `l_e`（soft 拓扑），两路通过
`pos_t = FK(θ_t, P+δ)` 耦合：节点深度越准 → 运动越干净 → rigidity 越锐 → 拓扑越对。

```
L = L_static(多视图 canonical RGB)
      + w_video·L_video(单目运动视频 RGB)        → 驱动 δ（深度）
      + L_topo(短边稀疏 + 度数≈树先验 + w_rigidity·L_rigid)  → 驱动 l_e（拓扑）
      + 0.5·L_struct_smooth(δ 沿边平滑)
```

**关键 fix（解耦）**：`L_topo` 里的几何量（边长、strain）**必须 detach**。否则其梯度反流进
δ，优化器会靠**移动节点假装边变短/刚性**来降 loss，把深度搞坏（实测两路一起炸，深度漂到
33cm）。detach 后 `L_topo` 只驱动 `l_e`、δ 只由 RGB 驱动，两路干净收敛。

### 8.5 实测结果（newplant9，纯深度扰动 3.05cm + 候选边拓扑，400 步）
| | node depth RMSE | topo AUC | topo prec@\|E*\| |
|---|---|---|---|
| motion_out（单目静态 + 仅几何） | 2.80cm（停滞） | 0.766 | 0.582 |
| **motion_in（+ 视频 + rigidity）** | **1.96cm（−30%）** | **0.997** | **0.955** |

深度单调收敛、拓扑 100 步内冲到 ~1.0；motion_out 两项都停滞（单目静态既看不到深度、也只能
靠边长猜连接）。图：`outputs/per_scene_optim/fuse/joint_verdict.png`。

### 8.6 组件与脚本
- `models/structure/edge_binding.py`（几何 binding + 叶片刚体）、`strpr_builder.py`
  （从 pretrain checkpoint 用 sklearn kmeans + PCA 重建 StPr）。
- `scripts/poc/synth_poc.py`（深度恢复 PoC）、`scripts/poc/synth_poc_topology.py`（rigidity 拓扑 PoC）、
  `scripts/pipeline/fuse_motion_structure.py`（集成联合优化器）+ 各自 viz。

### 8.7 已知边界 / 下一步
1. 生成视频不可直接用 → 落地真实结构需忠实拍摄视频。
2. 深度恢复幅度受**叶片遮挡**限制（branch 节点多被叶片挡住，RGB 梯度弱），叶片刚体 binding
   缓解但未根治。
3. FK 用固定初始树；拓扑是独立 re-estimation（rigidity 排序），未真正闭环改 FK 树结构。
4. `simulation/articulated_chain.py` 前向动力学对高植株**数值不稳定**（运动炸到数米，与力/刚度
   无关，源于全节点风力累积扭矩 = N×力×力臂）→ PoC 用规定的有界运动学摆动 θ_t + FK 绕开；
   真做物理识别需先稳定积分器（implicit / 更小 dt / per-node force 归一）。
