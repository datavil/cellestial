# definition of guides param in gggrid 
guides : str, default=’auto’

Specifies how guides (legends and colorbars) should be treated in the layout.

‘collect’ - collect guides from all subplots, removing duplicates.
‘keep’ - keep guides in their original subplots; do not collect at this level.
‘auto’ - allow guides to be collected if an upper-level layout uses guides='collect'; otherwise, keep them in subplots.
Duplicates are identified by comparing visual properties:
For legends: title, labels, and all aesthetic values (colors, shapes, sizes, etc.).
For colorbars: title, domain limits, breaks, and color gradient.