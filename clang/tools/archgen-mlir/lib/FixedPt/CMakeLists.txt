add_mlir_dialect_library(ArchGenFixedPt
FixedPt.cpp

ADDITIONAL_HEADER_DIRS

DEPENDS
ArchGenFixedPtIncGen

LINK_LIBS PUBLIC
MLIRIR
)
