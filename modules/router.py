from fastapi import APIRouter, status
from .scheme import LDAIn, LDAOut



modules_router = APIRouter()


@modules_router.post(
    '/',
    summary='LDA fit transform',
    status_code=status.HTTP_200_OK,
    response_model=LDAOut
)
async def lda_fit(
        lda_scheme: LDAIn
):
    pass