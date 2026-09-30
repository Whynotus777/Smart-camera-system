"""T14 data factory: downloads, fake cameras, generated clips, pre-labels, dedup.

Heavy or network-bound tools live here rather than in `src/scs/` so the product
package stays small. Nothing in `src/scs/` may import `data_ops`.
"""
