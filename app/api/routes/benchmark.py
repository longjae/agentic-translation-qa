from fastapi import APIRouter, Depends

from app.api.dependencies import get_benchmark_service
from app.schemas.benchmark import BenchmarkRequest, BenchmarkResponse
from app.services.benchmark import BenchmarkService

router = APIRouter()


@router.post("", response_model=BenchmarkResponse)
async def run_benchmark(
    request: BenchmarkRequest,
    service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkResponse:
    return await service.run(
        dataset_name=request.dataset_name,
        sample_size=request.sample_size,
        domain_filter=request.domain_filter,
    )
