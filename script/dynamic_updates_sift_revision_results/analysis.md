# SIFT dynamic-update analysis

All measured batches/checkpoints retained.
Insertion includes resize; query means/SD use three rounds. Footprint is serialized size.

## repeated
- Total insertion ratio: 2.072263x; per-batch range 1.903046--2.224793x.
- Final serialized-size reduction: 54.123%; retained deleted nodes: 600000.
- Mean delete cost raw/LVC: 0.5733/0.6885 us; LVC encoding share: 1.373%.
- tls_off: max recall deficit 0.280 pp; final recall raw/LVC 95.160/95.290%; final QPS 4574.57/601.42; latency ratios 6.271--7.610x.
- tls_0.2: max recall deficit 1.560 pp; final recall raw/LVC 95.160/93.670%; final QPS 4574.57/2033.37; latency ratios 1.935--2.250x.

## growing
- Total insertion ratio: 1.983412x; per-batch range 1.823005--2.089634x.
- Final serialized-size reduction: 54.192%; retained deleted nodes: 0.
- Mean delete cost raw/LVC: 0.0000/0.0000 us; LVC encoding share: 1.891%.
- tls_off: max recall deficit 0.240 pp; final recall raw/LVC 95.020/95.090%; final QPS 6173.15/908.72; latency ratios 6.232--7.073x.
- tls_0.2: max recall deficit 1.030 pp; final recall raw/LVC 95.020/94.570%; final QPS 6173.15/3044.89; latency ratios 1.797--2.132x.

Strict insertion <=2x at every batch: False.
No claims about recall ceilings: measurements use fixed efSearch, not an efSearch sweep.
Returned new version labels are mapped to source IDs by the benchmark before GT comparison.
