"""
modules/achievements_and_reports.py
-----------------------------------
Achievements System & Monthly Report Generator

1. Achievements: Track and award achievements based on spending patterns
2. Monthly Reports: Generate human-readable monthly summaries
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from modules.smart_insights_core import FeatureSet, Signal


# ═══════════════════════════════════════════════════════════════════════════════
# ── ACHIEVEMENTS SYSTEM ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Achievement:
    title: str
    description: str
    icon: str
    earned: bool = False
    earned_date: str | None = None


@dataclass
class AchievementResult:
    new_achievements: list[dict]
    all_achievements: list[dict]


ACHIEVEMENT_DEFINITIONS = {
    "high_spender": {
        "title": "High Spender 🔥",
        "description": "أنفق أكثر من 90% من ميزانية الاستمتاع في أول 15 يوم",
        "icon": "🔥",
    },
    "budget_risk": {
        "title": "Budget Risk ⚠️",
        "description": "تجاوزت مصاريف الاستمتاع 100% من الميزانية",
        "icon": "⚠️",
    },
    "saver_mode": {
        "title": "Saver Mode 💎",
        "description": "وفر 20% أو أكثر من الدخل الشهري",
        "icon": "💎",
    },
    "balanced_spender": {
        "title": "Balanced Spender ⚖️",
        "description": "حافظ على توازن بين الغذاء والاستمتاع",
        "icon": "⚖️",
    },
    "savings_master": {
        "title": "Savings Master 💰",
        "description": "حقق نسبة ادخار 20% أو أكثر لمدة 3 أشهر متتالية",
        "icon": "💰",
    },
    "budget_ninja": {
        "title": "Budget Ninja 🥷",
        "description": "حافظ على المصروفات الثابتة تحت السيطرة لمدة 6 أشهر",
        "icon": "🥷",
    },
    "smart_shopper": {
        "title": "Smart Shopper 🛒",
        "description": "خفض إنفاق الغذاء 15% أو أكثر مقارنة بالشهر السابق",
        "icon": "🛒",
    },
    "first_save": {
        "title": "First Save 🌱",
        "description": "سجل أول ادخار في النظام",
        "icon": "🌱",
    },
    "balanced_month": {
        "title": "Balanced Month ⚖️",
        "description": "حافظ على توازن بين جميع الفئات للشهر كامل",
        "icon": "⚖️",
    },
    "cost_cutter": {
        "title": "Cost Cutter ✂️",
        "description": "خفض المصروفات الثابتة عن الشهر السابق",
        "icon": "✂️",
    },
    "early_bird": {
        "title": "Early Bird 🌅",
        "description": "أنفق أقل من 80% من الميزانية في أول 10 أيام",
        "icon": "🌅",
    },
    "month_master": {
        "title": "Month Master 🏆",
        "description": "أتم الشهر دون تجاوز أي فئة",
        "icon": "🏆",
    },
}


class AchievementsSystem:
    def __init__(self, history: list[dict] | None = None):
        self.history = history or []
        self.earned_achievements: set[str] = set()
        self.new_achievements: list[dict] = []
    
    def check_achievements(
        self,
        features: FeatureSet,
        signals: list[Signal],
        fixed_expenses: dict[str, float],
    ) -> AchievementResult:
        """
        Check all achievements and return new ones.
        """
        self.new_achievements = []
        
        # Check new achievements
        self._check_high_spender(features)
        self._check_budget_risk(features)
        self._check_saver_mode(features)
        self._check_balanced_spender(features)
        
        # Check existing achievements
        self._check_savings_master(features)
        self._check_budget_ninja()
        self._check_smart_shopper(features)
        self._check_first_save(features)
        self._check_balanced_month(signals)
        self._check_cost_cutter()
        self._check_early_bird(features)
        self._check_month_master(features, signals)
        
        return AchievementResult(
            new_achievements=self.new_achievements,
            all_achievements=self.get_all_achievements(),
        )
    
    def _check_high_spender(self, features: FeatureSet) -> None:
        """Check if spent > 90% of optional budget in first 15 days."""
        if features.day > 15:
            return
        
        if features.optional_plan <= 0:
            return
        
        ratio = features.optional_spent / features.optional_plan
        if ratio >= 0.9:
            self._award("high_spender")
    
    def _check_budget_risk(self, features: FeatureSet) -> None:
        """Check if optional spending > 100% of budget."""
        if features.optional_plan <= 0:
            return
        
        if features.optional_spent > features.optional_plan:
            self._award("budget_risk")
    
    def _check_saver_mode(self, features: FeatureSet) -> None:
        """Check if saving ratio >= 20%."""
        if features.saving_ratio >= 0.2:
            self._award("saver_mode")
    
    def _check_balanced_spender(self, features: FeatureSet) -> None:
        """Check if balanced between food and optional."""
        if features.optional_plan <= 0 or features.food_plan <= 0:
            return
        
        food_ratio = features.food_spent / features.food_plan
        opt_ratio = features.optional_spent / features.optional_plan
        
        diff = abs(food_ratio - opt_ratio)
        if diff <= 0.2 and features.total_spent > 0:
            self._award("balanced_spender")
    
    def _check_savings_master(self, features: FeatureSet) -> None:
        """Check if saving >= 20% for 3 consecutive months."""
        if len(self.history) < 3:
            return
        
        consecutive_high_saving = 0
        for month_data in self.history[-3:]:
            saving_ratio = month_data.get("saving_ratio", 0)
            if saving_ratio >= 20:
                consecutive_high_saving += 1
            else:
                break
        
        if consecutive_high_saving >= 3:
            self._award("savings_master")
    
    def _check_budget_ninja(self) -> None:
        """Check if fixed expenses controlled for 6 months."""
        if len(self.history) < 6:
            return
        
        months_controlled = 0
        for month_data in self.history[-6:]:
            fixed_ok = month_data.get("fixed_ok", False)
            if fixed_ok:
                months_controlled += 1
            else:
                break
        
        if months_controlled >= 6:
            self._award("budget_ninja")
    
    def _check_smart_shopper(self, features: FeatureSet) -> None:
        """Check if food spending reduced by 15% vs previous month."""
        if len(self.history) < 1:
            return
        
        prev_food = self.history[-1].get("food_spent", 0)
        current_food = features.food_spent
        
        if prev_food > 0:
            reduction = ((prev_food - current_food) / prev_food) * 100
            if reduction >= 15:
                self._award("smart_shopper")
    
    def _check_first_save(self, features: FeatureSet) -> None:
        """Check if first saving recorded."""
        if features.saving_ratio >= 5:
            self._award("first_save")
    
    def _check_balanced_month(self, signals: list[Signal]) -> None:
        """Check if month was balanced."""
        signal_names = [s.name for s in signals]
        if "balanced_behavior" in signal_names:
            self._award("balanced_month")
    
    def _check_cost_cutter(self) -> None:
        """Check if fixed expenses reduced vs previous month."""
        if len(self.history) < 1:
            return
        
        prev_fixed = self.history[-1].get("fixed_total", 0)
        current_fixed = self.history[-1].get("current_fixed", 0) if self.history else 0
        
        if current_fixed > 0 and prev_fixed > current_fixed:
            reduction = ((prev_fixed - current_fixed) / prev_fixed) * 100
            if reduction >= 10:
                self._award("cost_cutter")
    
    def _check_early_bird(self, features: FeatureSet) -> None:
        """Check if spent < 80% of budget in first 10 days."""
        if features.day != 10:
            return
        
        total_plan = features.food_plan + features.optional_plan
        total_spent = features.food_spent + features.optional_spent
        
        if total_plan > 0 and total_spent <= total_plan * 0.8:
            self._award("early_bird")
    
    def _check_month_master(self, features: FeatureSet, signals: list[Signal]) -> None:
        """Check if completed month without any category overspend."""
        if features.day < 30:
            return
        
        critical_signals = [s for s in signals if s.severity == "critical"]
        if len(critical_signals) == 0:
            self._award("month_master")
    
    def _award(self, achievement_id: str) -> None:
        """Award an achievement if not already earned."""
        if achievement_id in self.earned_achievements:
            return
        
        if achievement_id not in ACHIEVEMENT_DEFINITIONS:
            return
        
        self.earned_achievements.add(achievement_id)
        definition = ACHIEVEMENT_DEFINITIONS[achievement_id]
        
        self.new_achievements.append({
            "id": achievement_id,
            "title": definition["title"],
            "description": definition["description"],
            "icon": definition["icon"],
            "earned_date": datetime.today().strftime("%Y-%m-%d"),
        })
    
    def _check_savings_master(self, features: FeatureSet) -> None:
        """Check if saving >= 20% for 3 consecutive months."""
        if len(self.history) < 3:
            return
        
        consecutive_high_saving = 0
        for month_data in self.history[-3:]:
            saving_ratio = month_data.get("saving_ratio", 0)
            if saving_ratio >= 20:
                consecutive_high_saving += 1
            else:
                break
        
        if consecutive_high_saving >= 3:
            self._award("savings_master")
    
    def _check_budget_ninja(self) -> None:
        """Check if fixed expenses controlled for 6 months."""
        if len(self.history) < 6:
            return
        
        months_controlled = 0
        for month_data in self.history[-6:]:
            fixed_ok = month_data.get("fixed_ok", False)
            if fixed_ok:
                months_controlled += 1
            else:
                break
        
        if months_controlled >= 6:
            self._award("budget_ninja")
    
    def _check_smart_shopper(self, features: FeatureSet) -> None:
        """Check if food spending reduced by 15% vs previous month."""
        if len(self.history) < 1:
            return
        
        prev_food = self.history[-1].get("food_spent", 0)
        current_food = features.food_spent
        
        if prev_food > 0:
            reduction = ((prev_food - current_food) / prev_food) * 100
            if reduction >= 15:
                self._award("smart_shopper")
    
    def _check_first_save(self, features: FeatureSet) -> None:
        """Check if first saving recorded."""
        if features.saving_ratio >= 5:
            self._award("first_save")
    
    def _check_balanced_month(self, signals: list[Signal]) -> None:
        """Check if month was balanced."""
        signal_names = [s.name for s in signals]
        if "balanced_behavior" in signal_names:
            self._award("balanced_month")
    
    def _check_cost_cutter(self) -> None:
        """Check if fixed expenses reduced vs previous month."""
        if len(self.history) < 1:
            return
        
        prev_fixed = self.history[-1].get("fixed_total", 0)
        current_fixed = self.history[-1].get("current_fixed", 0) if self.history else 0
        
        if current_fixed > 0 and prev_fixed > current_fixed:
            reduction = ((prev_fixed - current_fixed) / prev_fixed) * 100
            if reduction >= 10:
                self._award("cost_cutter")
    
    def _check_early_bird(self, features: FeatureSet) -> None:
        """Check if spent < 80% of budget in first 10 days."""
        if features.day != 10:
            return
        
        total_plan = features.food_plan + features.optional_plan
        total_spent = features.food_spent + features.optional_spent
        
        if total_plan > 0 and total_spent <= total_plan * 0.8:
            self._award("early_bird")
    
    def _check_month_master(self, features: FeatureSet, signals: list[Signal]) -> None:
        """Check if completed month without any category overspend."""
        if features.day < 30:
            return
        
        critical_signals = [s for s in signals if s.severity == "critical"]
        if len(critical_signals) == 0:
            self._award("month_master")
    
    def get_all_achievements(self) -> list[dict]:
        """Get all achievements with earned status."""
        result = []
        for aid, definition in ACHIEVEMENT_DEFINITIONS.items():
            result.append({
                "id": aid,
                "title": definition["title"],
                "description": definition["description"],
                "icon": definition["icon"],
                "earned": aid in self.earned_achievements,
            })
        return result


def generate_achievements(
    features: FeatureSet,
    signals: list[Signal],
    fixed_expenses: dict[str, float],
    history: list[dict] | None = None,
) -> AchievementResult:
    """
    Generate achievements based on current month and history.
    Returns new achievements and all achievements.
    """
    system = AchievementsSystem(history)
    return system.check_achievements(features, signals, fixed_expenses)


# ═══════════════════════════════════════════════════════════════════════════════
# ── MONTHLY REPORT GENERATOR ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MonthlyReport:
    stats: dict
    narrative: str
    insights: list[str]
    warnings: list[str]
    strengths: list[str]


REPORT_TEMPLATES = {
    "excellent_saving": [
        "أداؤك في الادخار ممتاز هذا الشهر! وفرت {ratio:.0f}% من دخلك.",
        "تحية إعجاب! نسبة الادخار {ratio:.0f}% تعكس انضباط مالي رائع.",
        "ميزانيتك تعمل بكفاءة عالية. وفرت {amount:,.0f} ج.م.",
    ],
    "good_saving": [
        "أداء جيد في الادخار. وفرت نسبة {ratio:.0f}% من دخلك.",
        "مستمر على المسار الصحيح. نسبة الادخار {ratio:.0f}% جيدة.",
    ],
    "low_saving": [
        "نسبة الادخار منخفضة هذا الشهر ({ratio:.0f}%). حاول تخصيص مبلغ ثابت الادخار.",
        "انتبه لادخارك! وفرت فقط {ratio:.0f}% من دخلك.",
    ],
    "high_food": [
        "إنفاق الغذاء مرتفع ({ratio:.0f}% من الميزانية). راجع عادات التسوق.",
        "الطعام يستهلك نسبة كبيرة من الميزانية ({ratio:.0f}%).",
    ],
    "high_enjoyment": [
        "مصاريف الاستمتاع تجاوزت المخطط. راجع قهوتك وتسوقك.",
        "الاستمتاع استهلك نسبة {ratio:.0f}% من الميزانية.",
    ],
    "fixed_controlled": [
        "المصروفات الثابتة تحت السيطرة. إدارة مالية مستقرة.",
        "فواتيرك الثابتة جيدة هذا الشهر.",
    ],
    "fixed_high": [
        "المصروفات الثابتة مرتفعة. راجع الفواتير.",
        "انتبه للفواتير الثابتة!",
    ],
    "balanced": [
        "ميزانيتك متوازنة بين الفئات المختلفة.",
        "أداء متناسق في جميع جوانب الإنفاق.",
    ],
}


class MonthlyReportGenerator:
    def __init__(self, features: FeatureSet, signals: list[Signal]):
        self.features = features
        self.signals = signals
        self.strengths: list[str] = []
        self.weaknesses: list[str] = []
        self.warnings: list[str] = []
        self.insights: list[str] = []
    
    def generate(self) -> MonthlyReport:
        """Generate full monthly report."""
        self._analyze_saving()
        self._analyze_food()
        self._analyze_enjoyment()
        self._analyze_fixed()
        self._analyze_balance()
        self._generate_warnings()
        
        narrative = self._build_narrative()
        stats = self._build_stats()
        
        return MonthlyReport(
            stats=stats,
            narrative=narrative,
            insights=self.insights,
            warnings=self.warnings,
            strengths=self.strengths,
        )
    
    def _analyze_saving(self) -> None:
        """Analyze saving performance."""
        f = self.features
        ratio = f.saving_ratio * 100
        
        if ratio >= 20:
            template = random.choice(REPORT_TEMPLATES["excellent_saving"])
            self.insights.append(template.format(ratio=ratio, amount=f.saving))
            self.strengths.append(f"ادخار ممتاز ({ratio:.0f}%)")
        elif ratio >= 10:
            template = random.choice(REPORT_TEMPLATES["good_saving"])
            self.insights.append(template.format(ratio=ratio))
            self.strengths.append(f"ادخار جيد ({ratio:.0f}%)")
        elif ratio < 5 and f.income > 0:
            template = random.choice(REPORT_TEMPLATES["low_saving"])
            self.insights.append(template.format(ratio=ratio))
            self.weaknesses.append(f"ادخار منخفض ({ratio:.0f}%)")
    
    def _analyze_food(self) -> None:
        """Analyze food spending."""
        f = self.features
        ratio = f.food_ratio * 100
        
        if ratio >= 90:
            template = random.choice(REPORT_TEMPLATES["high_food"])
            self.insights.append(template.format(ratio=ratio))
            self.weaknesses.append(f"إنفاق غذاء مرتفع ({ratio:.0f}%)")
        elif f.burn_rate_food < 0.8 and f.food_spent > 0:
            self.strengths.append("تحكم جيد في إنفاق الغذاء")
    
    def _analyze_enjoyment(self) -> None:
        """Analyze enjoyment spending."""
        f = self.features
        ratio = f.optional_ratio * 100
        
        if ratio >= 85:
            template = random.choice(REPORT_TEMPLATES["high_enjoyment"])
            self.insights.append(template.format(ratio=ratio))
            self.weaknesses.append(f"استمتاع مرتفع ({ratio:.0f}%)")
        
        if f.optional_overspend > 0:
            self.warnings.append(
                f"تجاوزت ميزانية الاستمتاع بـ {f.optional_overspend:,.0f} ج.م"
            )
    
    def _analyze_fixed(self) -> None:
        """Analyze fixed expenses."""
        f = self.features
        
        if f.fixed_deviation > 0.05:
            template = random.choice(REPORT_TEMPLATES["fixed_high"])
            self.insights.append(template)
            self.weaknesses.append("مصروفات ثابتة مرتفعة")
        elif f.fixed_deviation < 0 and f.fixed_total > 0:
            template = random.choice(REPORT_TEMPLATES["fixed_controlled"])
            self.insights.append(template)
            self.strengths.append("مصروفات ثابتة تحت السيطرة")
    
    def _analyze_balance(self) -> None:
        """Analyze overall balance."""
        signal_names = [s.name for s in self.signals]
        
        if "balanced_behavior" in signal_names:
            template = random.choice(REPORT_TEMPLATES["balanced"])
            self.insights.append(template)
            self.strengths.append("ميزانية متوازنة")
    
    def _generate_warnings(self) -> None:
        """Generate warnings from critical signals."""
        for signal in self.signals:
            if signal.severity == "critical":
                if signal.name == "fast_food_spending":
                    self.warnings.append(
                        f"إنفاق الغذاء يتسارع! ({signal.value:.0f}% من المتوقع)"
                    )
                elif signal.name == "fast_optional_spending":
                    self.warnings.append(
                        f"إنفاق الاستمتاع يتسارع! ({signal.value:.0f}% من المتوقع)"
                    )
                elif signal.name == "saving_at_risk":
                    self.warnings.append("الادخار في خطر هذا الشهر!")
    
    def _build_stats(self) -> dict:
        """Build stats summary."""
        f = self.features
        return {
            "total_spent": f.total_spent,
            "food_spent": f.food_spent,
            "optional_spent": f.optional_spent,
            "fixed_spent": f.fixed_total,
            "saving": f.saving,
            "saving_ratio": round(f.saving_ratio * 100, 1),
            "food_ratio": round(f.food_ratio * 100, 1),
            "optional_ratio": round(f.optional_ratio * 100, 1),
            "day": f.day,
            "days_left": f.days_left,
        }
    
    def _build_narrative(self) -> str:
        """Build human-readable narrative."""
        f = self.features
        
        # Opening
        if f.day <= 10:
            opening = f"بداية الشهر جيدة! أنت في اليوم {f.day}."
        elif f.day <= 20:
            opening = f"نحن في منتصف الشهر ({f.day}/30)."
        else:
            opening = f"الأشهر يوشك على الانتهاء. بقى {f.days_left} يوم."
        
        # Main body
        parts = []
        
        if self.strengths:
            strength_text = " • ".join(self.strengths[:2])
            parts.append(f"نقاط القوة: {strength_text}")
        
        if self.weaknesses:
            weakness_text = " • ".join(self.weaknesses[:2])
            parts.append(f"نقاط تحتاج تحسين: {weakness_text}")
        
        # Saving summary
        if f.saving > 0:
            parts.append(f"صافي الادخار: {f.saving:,.0f} ج.م ({f.saving_ratio*100:.0f}%)")
        else:
            parts.append("تحتاج مراجعة المصروفات للحفاظ على الادخار.")
        
        # Advice
        advice = ""
        if f.optional_overspend > 0:
            advice = "راجع مصاريف الاستمتاع للبقاء على المسار."
        elif f.saving_ratio < 0.1:
            advice = "حاول زيادة نسبة الادخار لهذا الشهر."
        elif f.saving_ratio >= 0.2:
            advice = "أداء ممتاز! استمر على هذا المنوال."
        
        if advice:
            parts.append(f"النصيحة: {advice}")
        
        return opening + " " + ". ".join(parts)


def generate_monthly_report(
    features: FeatureSet,
    signals: list[Signal],
) -> MonthlyReport:
    """
    Generate monthly report from features and signals.
    """
    generator = MonthlyReportGenerator(features, signals)
    return generator.generate()