# 第一阶段收口单：Linux 上冻结 C3 基准

状态日期：2026-09-15

## 本阶段唯一目标

在 Linux 服务器上完成 MMseqs2 30% 同源聚类并原子冻结 `c3_primary`。
本阶段不安装或运行 Protenix，不生成 B1/B2/B3 结果，也不开始结构微调。

## 已核实的本地起点

- 输入目录：`protenix_ppi/data/processed/human_ppi_2026_03_v0.3`
- `proteins.csv` SHA-256：`aec9b7e9c34051daad82a145c317622a804b9d498171b6ff466d27228d1c5840`
- `mmseqs30/proteins.clustered.csv`：尚不存在
- `splits/c3_primary/splits.csv`：尚不存在
- 本地回归：157 项测试通过

先在服务器上校验传输后的输入哈希：

```bash
sha256sum protenix_ppi/data/processed/human_ppi_2026_03_v0.3/proteins.csv
```

输出必须与上面的 SHA-256 完全一致，否则停止。

## 只执行以下两条命令

从包含 `protenix_ppi/` 的目录运行：

```bash
bash protenix_ppi/scripts/run_mmseqs2_clustering.sh \
  protenix_ppi/data/processed/human_ppi_2026_03_v0.3/proteins.csv \
  protenix_ppi/data/processed/human_ppi_2026_03_v0.3/mmseqs30 \
  8

python3 -m protenix_ppi.scripts.finalize_c3_benchmark \
  --dataset-dir protenix_ppi/data/processed/human_ppi_2026_03_v0.3
```

脚本固定使用：`--min-seq-id 0.30`、`-c 0.50`、`--cov-mode 0`、
`--alignment-mode 3`、`--cluster-mode 0`、`-s 7.5`。不要临时修改参数。

## 收口验收

以下四项同时成立，本阶段才算完成：

1. 两条命令退出码均为 0。
2. `mmseqs30/proteins.clustered.csv` 与 `mmseqs30/mmseqs_cluster_metadata.json` 存在。
3. `splits/c3_primary/splits.csv` 与 `benchmark_freeze_manifest.json` 存在。
4. `c3_validation.json` 中没有 error，且冻结脚本没有报告数据集、蛋白、同源簇或复合物泄漏。

完成后保留整个数据目录及终端日志，不要重新生成 split。下一阶段再单独进行
Linux G0/G1；在本收口单验收前不启动真实 Protenix 实验。
