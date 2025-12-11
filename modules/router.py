from fastapi import APIRouter, status
from .scheme import LDAIn, LDAOut
from lda import LDA


modules_router = APIRouter()

linear_da = LDA(2)

@modules_router.post(
    '/',
    summary='LDA fit transform',
    status_code=status.HTTP_200_OK,
    response_model=LDAOut
)
async def lda_fit(
        lda_scheme: LDAIn
) -> LDAOut:
    lda_scheme = LDAOut(
        X=linear_da.fit(lda_scheme.X, lda_scheme.y).tolist(),
        y=lda_scheme.y.tolist()
    )
    return lda_scheme