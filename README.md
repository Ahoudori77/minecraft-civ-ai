### Dependency Pinning

| Component | Version | 理由 |
|-----------|---------|------|
| gym       | **0.19** | MineRL 0.4.4 がハード依存 |
| cleanrl   | **0.4.8** | gym 0.19 系と互換。0.6.x は gymnasium 前提のため未対応 |
| SB3       | **1.8.0** | cleanrl 0.4.x とペアになる推奨バージョン |
