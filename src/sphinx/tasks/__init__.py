# sphinx/tasks/__init__.py
from .symmetry import (
    SymmetryGridMirrorFillTask,
    SymmetrySceneMirrorIdentifyTask,
    SymmetryFriezeGroupsTask,
    SymmetryWallpaperGroupsTask,
)
from .sequence import SequenceArithmeticTask, SequenceRotationTask, SequenceMultiColumnArithmeticTask
from .transform import (
    TransformPairInferTask,
    TransformResultIdentifyTask,
    TransformSimilarityIdentifyTask,
)
from .tiles import (
    TilesConnectedComponentTask,
    TilesShortestPathTask,
    TilesMissingTilesTask,
    TilesDecomposeComposeTask,
    TilesGeometryTask,
    TilesRecoloringTask,
    TilesLineLengthTask,
    TilesLineIntersectionsTask,
)

from .counting import ShapeCountTask

from .charts import ChartsPieTask, ChartsMatchProportionsTask

from .geometric import GeometricSortTask, GeometricPositionTask, GeometricStackCountTask


from .rect_venn import RectVennTask

__all__ = [
    "SymmetryGridMirrorFillTask",
    "SymmetrySceneMirrorIdentifyTask",
    "SymmetryFriezeGroupsTask",
    "SymmetryWallpaperGroupsTask",
    "SequenceArithmeticTask",
    "SequenceRotationTask",
    "SequenceMultiColumnArithmeticTask",
    "TransformPairInferTask",
    "TransformResultIdentifyTask",
    "TransformSimilarityIdentifyTask",
    "TilesConnectedComponentTask",
    "TilesShortestPathTask",
    "TilesMissingTilesTask",
    "TilesDecomposeComposeTask",
    "TilesGeometryTask",
    "TilesLineLengthTask",
    "TilesLineIntersectionsTask",
    "TilesRecoloringTask",
    "ShapeCountTask",
    "ChartsPieTask",
    "ChartsMatchProportionsTask",
    "GeometricPositionTask",
    "GeometricSortTask",
    "GeometricStackCountTask",
    "RectVennTask",
]
