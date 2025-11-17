"""
Database Schemas for World Truffle Index (WTI)

Each Pydantic model maps to a MongoDB collection (lowercased class name).
"""
from __future__ import annotations
from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import Optional, List, Dict, Literal
from datetime import datetime

TruffleType = Literal[
    "white",        # Tuber magnatum
    "black",        # Tuber melanosporum
    "summer",       # Tuber aestivum
    "uncinatum",    # Tuber uncinatum
    "bianchetto",   # Tuber borchii
    "others"
]

Region = Literal[
    "italy",
    "france",
    "spain",
    "australia",
    "us",
    "middle_east"
]

class Supplier(BaseModel):
    name: str = Field(..., description="Company or hunter name")
    email: Optional[EmailStr] = None
    region: Optional[Region] = None
    verified: bool = Field(False, description="Whether supplier is verified")

class TruffleObservation(BaseModel):
    truffle_type: TruffleType
    region: Region
    price_per_kg: float = Field(..., gt=0, description="Price per kg in EUR")
    quantity_kg: float = Field(1.0, gt=0, description="Quantity represented by this observation")
    # Quality sub-metrics (0-10 except size_mm and freshness_days)
    size_mm: float = Field(..., gt=0, description="Max diameter in millimeters")
    aroma_score: float = Field(..., ge=0, le=10)
    freshness_days: float = Field(..., ge=0, description="Days since harvest")
    color_score: float = Field(..., ge=0, le=10)
    texture_score: float = Field(..., ge=0, le=10)
    origin_certified: bool = Field(False)
    handling_score: float = Field(..., ge=0, le=10)

    supply_level: Literal["very_low","low","balanced","high","very_high"] = "balanced"
    demand_level: Literal["very_low","low","balanced","high","very_high"] = "balanced"
    volatility_score: float = Field(5.0, ge=0, le=10)

    week: str = Field(..., description="ISO week, e.g., 2025-W02")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: Optional[str] = Field(None, description="Supplier name, market, marketplace, etc.")

class WTIIndex(BaseModel):
    week: str
    global_index: float
    by_type: Dict[str, float] = Field(default_factory=dict)
    by_region: Dict[str, float] = Field(default_factory=dict)
    by_type_region: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    # Market metrics
    supply_signal: Dict[str, float] = Field(default_factory=dict)  # 0-100
    demand_signal: Dict[str, float] = Field(default_factory=dict)  # 0-100
    volatility: Dict[str, float] = Field(default_factory=dict)     # 0-100
    forecast_next_week: Dict[str, float] = Field(default_factory=dict)  # % change forecast
    created_at: datetime = Field(default_factory=datetime.utcnow)

class QualityStandard(BaseModel):
    version: str = "1.0"
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "size_mm": 0.15,
            "aroma_score": 0.25,
            "freshness_days": 0.15,
            "color_score": 0.15,
            "texture_score": 0.15,
            "origin_certified": 0.05,
            "handling_score": 0.10,
        }
    )
    notes: Optional[str] = "Scores normalized to 0-100. Freshness uses decay function." 

class SeasonalityWindow(BaseModel):
    truffle_type: TruffleType
    start_month: int = Field(..., ge=1, le=12)
    end_month: int = Field(..., ge=1, le=12)
    peak_months: List[int] = Field(default_factory=list)
    regions: Optional[List[Region]] = None

# Example content models for weekly reports
class WeeklyReport(BaseModel):
    week: str
    highlights: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    commentary: Optional[str] = None
