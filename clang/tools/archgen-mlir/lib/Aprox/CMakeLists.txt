add_mlir_dialect_library(ArchGenAprox
Aprox.cpp

ADDITIONAL_HEADER_DIRS

DEPENDS
ArchGenFixedPt
ArchGenAproxIncGen

LINK_LIBS PUBLIC
ArchGenFixedPt
MLIRIR
)
