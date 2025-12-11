from pydantic import BaseModel, field_validator
from numpy import array, nan, isnan, nanmean, where, take
from fastapi import HTTPException, status


class LDAOut(BaseModel):

    X: list[list]
    y: list


class LDAIn(BaseModel):
    model_config = {
        'extra': 'forbid'
    }

    X: list[list]
    y: list

    @field_validator('X')
    def verify_X(cls, value):
        X = array([[nan if j is None else j for j in row] for row in value])
        if X.ndim != 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='X must be 2D matrix'
            )
        mean = nanmean(X, axis=0)
        idx = where(isnan(X))
        X[idx] = take(mean, idx[1])
        return X

    @field_validator('y')
    def verify_y(cls, value):
        y = array(value)
        if y.ndim != 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='y must be 1D vector'
            )
        if isnan(y).any():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='Target values should not contain NaN'
            )
        return y