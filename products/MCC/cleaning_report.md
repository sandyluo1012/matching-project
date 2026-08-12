# MCC 产品表清理报告

缺失值策略：删除数据区整列为空的字段；保留局部 NA，不生成推算值。匹配模型只比较双方均有值的参数，并计算覆盖率。

含逗号策略：如果一个物料的任意参数字段包含逗号，则删除整颗物料；通用信息字段不参与此规则。

| 产品类别 | 原始数量 | 清理后数量 | 删除物料 | 删除不可用列 | 保留空白 |
|---|---:|---:|---:|---:|---:|
| Bridge Rectifiers | 422 | 422 | 0 | 0 | 0 |
| Darlington Transistors | 11 | 11 | 0 | 0 | 11 |
| ESD Protection Devices | 544 | 543 | 1 | 0 | 234 |
| Fast Recovery Rectifiers | 105 | 105 | 0 | 1 | 0 |
| Medium Power Bipolar Transistors | 98 | 98 | 0 | 0 | 39 |
| Power MOSFETS | 716 | 716 | 0 | 0 | 1065 |
| Pre-Biased Transistors | 248 | 248 | 0 | 0 | 26 |
| Programmable Thyristor Surge Suppressor | 1 | 1 | 0 | 1 | 0 |
| Schottky Barrier Rectifiers | 760 | 760 | 0 | 0 | 672 |
| Small Signal Bipolar Transistors | 374 | 374 | 0 | 0 | 77 |
| Small Signal MOSFETS | 299 | 299 | 0 | 0 | 271 |
| Small Signal Schottky Diodes | 284 | 284 | 0 | 0 | 179 |
| Standard Recovery Rectifiers | 200 | 200 | 0 | 1 | 119 |
| Super Fast Recovery Rectifiers | 433 | 433 | 0 | 0 | 434 |
| Switching Diodes | 160 | 160 | 0 | 0 | 119 |
| TVS | 4681 | 4681 | 0 | 0 | 66 |
| Wide SOA MOSFETs | 3 | 3 | 0 | 4 | 0 |
| Zener Diodes | 2144 | 2144 | 0 | 1 | 185 |

## 删除的全空或未命名列

- **Fast Recovery Rectifiers**：`Configuration`
- **Programmable Thyristor Surge Suppressor**：`Configuration`
- **Standard Recovery Rectifiers**：`Configuration`
- **Wide SOA MOSFETs**：`RDS(ON) Max @VGS=4.5V (Ω)`、`RDS(ON) Max @VGS=2.5V (Ω)`、`未命名列 18`、`未命名列 19`
- **Zener Diodes**：`Configuration`

## 被删除的物料

- **ESD Protection Devices / ESD0512LB**
  - `Reverse Standoff Voltage VRWM(V)`：`12, 5`
  - `Breakdown Voltage Min VBR(V)`：`13, 6`
  - `Breakdown Voltage Max VBR(V)`：`17, 10`

## 保留的局部空白

这些值保持未知，不会被中位数或近邻值替代。

- **Darlington Transistors**：HFE [max]：7；fT(MHz)：4
- **ESD Protection Devices**：Junction Capacitance CJ(pF)：4；Peak Pluse Power Dissipation PPPK (W)：1；Breakdown Voltage Min VBR(V)：2；Breakdown Voltage Max VBR(V)：223；VESDIEC61000-4-2 Air/Contact (kV)：4
- **Medium Power Bipolar Transistors**：HFE [max]：30；fT(MHz)：9
- **Power MOSFETS**：Drain-Source On-Resistance RDS(ON) Max @VGS=10V (Ω)：10；Drain-Source On-Resistance RDS(ON) Max @VGS=4.5V (Ω)：326；Drain-Source On-Resistance RDS(ON) Max @VGS=2.5V (Ω)：705；Single Pulsed Avalanche Energy EAS(mJ)：24
- **Pre-Biased Transistors**：R2 Typ (KΩ)：25；fT(MHz)：1
- **Schottky Barrier Rectifiers**：Configuration：672
- **Small Signal Bipolar Transistors**：HFE [max]：67；fT(MHz)：10
- **Small Signal MOSFETS**：RDS(ON) Max @VGS=10V (Ω)：110；RDS(ON) Max @VGS=4.5V (Ω)：11；RDS(ON) Max @VGS=2.5V (Ω)：150
- **Small Signal Schottky Diodes**：Configuration：178；IF(AV)(A)：1
- **Standard Recovery Rectifiers**：TRR (μs)：119
- **Super Fast Recovery Rectifiers**：Configuration：431；VR (V)：3
- **Switching Diodes**：Configuration：110；TRR (μs)：5；PD (mW)：4
- **TVS**：Peak Pulse Power Dissipation PPPM (W)：40；Breakdown Voltage Max VBR(V)：26
- **Zener Diodes**：ZZT(Ω) @IZT：39；ZZK(Ω) @IZT：73；IZK(mA)：73
