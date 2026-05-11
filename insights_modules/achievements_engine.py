"""
modules/achievements_engine.py
------------------------------
User Achievements System for the Smart Budget System.

Tracks and awards achievements based on:
- Consecutive savings performance (3+ months)
- Fixed expenses stability (6+ months)
- Food spending reduction vs previous month
- Other behavioral milestones
"""

from __future__ import annotations

import sqlite3
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
os.makedirs(_DATA_DIR, exist_ok=True)
ACHIEVEMENTS_DB = os.path.join(_DATA_DIR, "achievements.db")


@dataclass
class Achievement:
    title: str
    description: str
    icon: str
    earned: bool = False
    earned_date: str | None = None


@dataclass
class AchievementNotification:
    title: str
    message: str
    icon: str
    achievement_type: str


AchievementType = Literal[
    "high_spender",
    "budget_risk",
    "saver_mode",
    "balanced_spender",
    "savings_master",
    "budget_ninja",
    "smart_shopper",
    "early_bird",
    "balance_master",
    "conservative_saver",
    "first_step",
    "steady_saver",
    "budget_tracker",
    "category_expert",
]

ACHIEVEMENT_DEFINITIONS: dict[AchievementType, dict] = {
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
        "description": "حقق هدف الادخار بنسبة 20% أو أكثر لمدة 3 أشهر متتالية",
        "icon": "💰",
    },
    "budget_ninja": {
        "title": "Budget Ninja 🥷",
        "description": "حافظ على المصروفات الثابتة تحت السيطرة لمدة 6 أشهر متتالية",
        "icon": "🥷",
    },
    "smart_shopper": {
        "title": "Smart Shopper 🛒",
        "description": "خفض إنفاق الطعام بنسبة 15% أو أكثر مقارنة بالشهر السابق",
        "icon": "🛒",
    },
    "early_bird": {
        "title": "Early Bird 🌅",
        "description": "أنفق أقل من 80% من الميزانية المخططة في أول 10 أيام",
        "icon": "🌅",
    },
    "balance_master": {
        "title": "Balance Master ⚖️",
        "description": "حافظ على توازن مثالي بين جميع فئات الإنفاق لمدة شهر كامل",
        "icon": "⚖️",
    },
    "conservative_saver": {
        "title": "Conservative Saver 🛡️",
        "description": "ادخر 25% أو أكثر من إجمالي الميزانية",
        "icon": "🛡️",
    },
    "first_step": {
        "title": "First Step 🌟",
        "description": "سجّل أول مصروف في النظام",
        "icon": "🌟",
    },
    "steady_saver": {
        "title": "Steady Saver 📈",
        "description": "ادخر بنفس المبلغ أو أكثر لمدة شهرين متتاليين",
        "icon": "📈",
    },
    "budget_tracker": {
        "title": "Budget Tracker 📊",
        "description": "تتبع ميزانيتك لمدة 3 أشهر متتالية",
        "icon": "📊",
    },
    "category_expert": {
        "title": "Category Expert 🎯",
        "description": "حافظ على فئة واحدة ضمن الميزانية لمدة 6 أشهر",
        "icon": "🎯",
    },
}


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(ACHIEVEMENTS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_achievements_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS achievements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                achievement_type TEXT NOT NULL,
                earned_date TEXT,
                metadata TEXT,
                UNIQUE(user_id, achievement_type)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS monthly_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                month TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                UNIQUE(user_id, month, category)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expense_count (
                user_id TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_month
            ON monthly_history(user_id, month)
        """)
        conn.commit()


def save_monthly_snapshot(
    user_id: str,
    month: str,
    spending: dict[str, float],
    plan: dict[str, float],
) -> None:
    with _get_conn() as conn:
        for category, amount in spending.items():
            planned = plan.get(category, 0.0)
            conn.execute("""
                INSERT OR REPLACE INTO monthly_history
                (user_id, month, category, amount)
                VALUES (?, ?, ?, ?)
            """, (user_id, month, category, amount))
        conn.commit()


def increment_expense_count(user_id: str) -> int:
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO expense_count (user_id, count)
            VALUES (?, 1)
            ON CONFLICT(user_id) DO UPDATE SET count = count + 1
        """, (user_id,))
        conn.commit()
        row = conn.execute("SELECT count FROM expense_count WHERE user_id = ?", (user_id,)).fetchone()
        return row["count"] if row else 0


def get_expense_count(user_id: str) -> int:
    with _get_conn() as conn:
        row = conn.execute("SELECT count FROM expense_count WHERE user_id = ?", (user_id,)).fetchone()
        return row["count"] if row else 0


def get_monthly_history(
    user_id: str,
    months: int = 12,
) -> list[dict]:
    cutoff = (datetime.today().replace(day=1) - timedelta(days=30 * months)).strftime("%Y-%m")
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT month, category, amount
            FROM monthly_history
            WHERE user_id = ? AND month >= ?
            ORDER BY month ASC
        """, (user_id, cutoff)).fetchall()
    history = {}
    for row in rows:
        month = row["month"]
        if month not in history:
            history[month] = {}
        history[month][row["category"]] = row["amount"]
    return [{"month": m, "spending": data} for m, data in sorted(history.items())]


def get_previous_month_spending(user_id: str, category: str) -> float | None:
    prev_month = (datetime.today().replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    with _get_conn() as conn:
        row = conn.execute("""
            SELECT amount FROM monthly_history
            WHERE user_id = ? AND month = ? AND category = ?
        """, (user_id, prev_month, category)).fetchone()
    return float(row["amount"]) if row else None


def has_achievement(user_id: str, achievement_type: AchievementType) -> bool:
    with _get_conn() as conn:
        row = conn.execute("""
            SELECT 1 FROM achievements
            WHERE user_id = ? AND achievement_type = ?
        """, (user_id, achievement_type)).fetchone()
    return row is not None


def award_achievement(
    user_id: str,
    achievement_type: AchievementType,
    metadata: dict | None = None,
) -> Achievement:
    earned_date = datetime.today().strftime("%Y-%m-%d")
    meta_str = str(metadata) if metadata else None
    with _get_conn() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO achievements
            (user_id, achievement_type, earned_date, metadata)
            VALUES (?, ?, ?, ?)
        """, (user_id, achievement_type, earned_date, meta_str))
        conn.commit()
    definition = ACHIEVEMENT_DEFINITIONS[achievement_type]
    return Achievement(
        title=definition["title"],
        description=definition["description"],
        icon=definition["icon"],
        earned=True,
        earned_date=earned_date,
    )


def get_all_user_achievements(user_id: str) -> list[Achievement]:
    achievements = []
    earned_types = set()
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT achievement_type, earned_date FROM achievements
            WHERE user_id = ?
        """, (user_id,)).fetchall()
        for row in rows:
            earned_types.add(row["achievement_type"])
    for atype, definition in ACHIEVEMENT_DEFINITIONS.items():
        achievements.append(Achievement(
            title=definition["title"],
            description=definition["description"],
            icon=definition["icon"],
            earned=atype in earned_types,
            earned_date=None,
        ))
    return achievements


class AchievementsEvaluator:
    def __init__(
        self,
        user_id: str,
        current_spending: dict[str, float],
        current_plan: dict[str, float],
        current_day: int,
    ):
        self.user_id = user_id
        self.current_spending = current_spending
        self.current_plan = current_plan
        self.current_day = current_day
        self.new_achievements: list[Achievement] = []
        self.notifications: list[AchievementNotification] = []

    def _save_current_month(self) -> None:
        month = datetime.today().strftime("%Y-%m")
        save_monthly_snapshot(self.user_id, month, self.current_spending, self.current_plan)

    def _increment_expenses(self) -> None:
        increment_expense_count(self.user_id)

    def evaluate(self) -> list[dict]:
        self._increment_expenses()
        self._save_current_month()
        self._check_first_step()
        
        # New achievements
        self._check_high_spender()
        self._check_budget_risk()
        self._check_saver_mode()
        self._check_balanced_spender()
        
        # Existing achievements
        self._check_savings_master()
        self._check_budget_ninja()
        self._check_smart_shopper()
        self._check_early_bird()
        self._check_balance_master()
        self._check_conservative_saver()
        self._check_steady_saver()
        self._check_budget_tracker()
        self._check_category_expert()
        return self.to_dict_list()

    def to_dict_list(self) -> list[dict]:
        return [
            {
                "title": a.title,
                "description": a.description,
                "icon": a.icon,
                "earned": a.earned,
                "earned_date": a.earned_date,
            }
            for a in self.new_achievements
        ]

    def get_notifications(self) -> list[dict]:
        return [
            {
                "title": n.title,
                "message": n.message,
                "icon": n.icon,
                "type": n.achievement_type,
            }
            for n in self.notifications
        ]

    def _award(self, achievement_type: AchievementType, with_notification: bool = True) -> None:
        if has_achievement(self.user_id, achievement_type):
            return
        achievement = award_achievement(self.user_id, achievement_type)
        self.new_achievements.append(achievement)
        if with_notification:
            definition = ACHIEVEMENT_DEFINITIONS[achievement_type]
            self.notifications.append(AchievementNotification(
                title=f"🏆 إنجاز جديد!",
                message=f'{definition["icon"]} {definition["title"]}\n{definition["description"]}',
                icon=definition["icon"],
                achievement_type=achievement_type,
            ))

    def _check_first_step(self) -> None:
        expense_count = get_expense_count(self.user_id)
        if expense_count >= 1:
            self._award("first_step")
    
    def _check_high_spender(self) -> None:
        """Check if spent > 90% of optional budget (enjoyment) in first 15 days."""
        if self.current_day > 15:
            return
        optional_plan = self.current_plan.get("enjoyment", 0)
        if optional_plan <= 0:
            return
        optional_subcats = {"coffee", "shopping", "entertainment", "other"}
        optional_spent = sum(
            self.current_spending.get(cat.lower(), 0) 
            for cat in optional_subcats
        )
        if optional_plan > 0 and optional_spent >= optional_plan * 0.9:
            self._award("high_spender")
    
    def _check_budget_risk(self) -> None:
        """Check if optional spending > 100% of budget."""
        optional_plan = self.current_plan.get("enjoyment", 0)
        if optional_plan <= 0:
            return
        optional_subcats = {"coffee", "shopping", "entertainment", "other"}
        optional_spent = sum(
            self.current_spending.get(cat.lower(), 0) 
            for cat in optional_subcats
        )
        if optional_spent > optional_plan:
            self._award("budget_risk")
    
    def _check_saver_mode(self) -> None:
        """Check if saving >= 20% of income."""
        income = getattr(self, 'current_income', 10000.0)
        savings = self.current_spending.get("saving", 0)
        if income > 0 and savings >= income * 0.2:
            self._award("saver_mode")
    
    def _check_balanced_spender(self) -> None:
        """Check if balanced between food and optional."""
        food_plan = self.current_plan.get("food", 0)
        optional_plan = self.current_plan.get("enjoyment", 0)
        if food_plan <= 0 or optional_plan <= 0:
            return
        food_spent = self.current_spending.get("food", 0)
        optional_subcats = {"coffee", "shopping", "entertainment", "other"}
        optional_spent = sum(
            self.current_spending.get(cat.lower(), 0) 
            for cat in optional_subcats
        )
        food_ratio = food_spent / food_plan
        opt_ratio = optional_spent / optional_plan
        diff = abs(food_ratio - opt_ratio)
        if diff <= 0.2 and (food_spent + optional_spent) > 0:
            self._award("balanced_spender")

    def _check_savings_master(self) -> None:
        history = get_monthly_history(self.user_id, months=6)
        if len(history) < 3:
            return
        savings_key = "savings"
        consecutive = 0
        for month_data in history[-3:]:
            spent = month_data["spending"].get(savings_key, 0)
            plan = self.current_plan.get(savings_key, 0)
            if plan > 0 and spent >= plan * 0.20:
                consecutive += 1
            else:
                break
        if consecutive >= 3:
            self._award("savings_master")

    def _check_budget_ninja(self) -> None:
        history = get_monthly_history(self.user_id, months=12)
        fixed_cats = {"rent", "electricity", "water", "gas", "internet", "mobile"}
        if len(history) < 6:
            return
        months_stable = 0
        for month_data in history[-6:]:
            spending = month_data["spending"]
            all_fixed_controlled = True
            for cat in fixed_cats:
                if cat in spending:
                    cat_spent = spending[cat]
                    cat_plan = self.current_plan.get(cat, cat_spent)
                    if cat_plan > 0 and cat_spent > cat_plan * 1.1:
                        all_fixed_controlled = False
                        break
            if all_fixed_controlled:
                months_stable += 1
        if months_stable >= 6:
            self._award("budget_ninja")

    def _check_smart_shopper(self) -> None:
        if len(self.current_spending) == 0:
            return
        prev_food = get_previous_month_spending(self.user_id, "food")
        if prev_food is None or prev_food == 0:
            return
        current_food = self.current_spending.get("food", 0)
        reduction_pct = ((prev_food - current_food) / prev_food) * 100
        if reduction_pct >= 15:
            self._award("smart_shopper")

    def _check_early_bird(self) -> None:
        if self.current_day != 10:
            return
        total_spent = sum(self.current_spending.values())
        total_plan = sum(self.current_plan.values())
        if total_plan > 0 and total_spent <= total_plan * 0.80:
            self._award("early_bird")

    def _check_balance_master(self) -> None:
        if len(self.current_spending) < 3:
            return
        all_normal = all(
            self.current_plan.get(cat, 0) == 0 or
            self.current_spending.get(cat, 0) <= self.current_plan.get(cat, 0) * 1.05
            for cat in self.current_spending
        )
        if all_normal and self.current_day >= 25:
            self._award("balance_master")

    def _check_conservative_saver(self) -> None:
        savings_key = "savings"
        savings_spent = self.current_spending.get(savings_key, 0)
        total_plan = sum(self.current_plan.values())
        if total_plan > 0:
            savings_ratio = savings_spent / total_plan
            if savings_ratio >= 0.25:
                self._award("conservative_saver")

    def _check_steady_saver(self) -> None:
        history = get_monthly_history(self.user_id, months=4)
        if len(history) < 2:
            return
        savings_key = "savings"
        consecutive = 0
        prev_amount = 0
        for month_data in history[-2:]:
            spent = month_data["spending"].get(savings_key, 0)
            if spent >= prev_amount and spent > 0:
                consecutive += 1
            else:
                break
            prev_amount = spent
        if consecutive >= 2:
            self._award("steady_saver")

    def _check_budget_tracker(self) -> None:
        history = get_monthly_history(self.user_id, months=12)
        if len(history) >= 3:
            self._award("budget_tracker")

    def _check_category_expert(self) -> None:
        history = get_monthly_history(self.user_id, months=12)
        if len(history) < 6:
            return
        fixed_cats = {"rent", "electricity", "water", "gas", "internet", "mobile"}
        for cat in fixed_cats:
            months_controlled = 0
            for month_data in history[-6:]:
                spending = month_data["spending"]
                if cat in spending:
                    cat_spent = spending[cat]
                    cat_plan = self.current_plan.get(cat, cat_spent)
                    if cat_plan > 0 and cat_spent <= cat_plan:
                        months_controlled += 1
                    else:
                        break
            if months_controlled >= 6:
                self._award("category_expert")
                break


def evaluate_achievements(
    user_id: str,
    current_spending: dict[str, float],
    current_plan: dict[str, float],
    current_day: int,
) -> list[dict]:
    evaluator = AchievementsEvaluator(user_id, current_spending, current_plan, current_day)
    return evaluator.evaluate()


def get_user_achievements(user_id: str) -> list[dict]:
    achievements = get_all_user_achievements(user_id)
    return [
        {
            "title": a.title,
            "description": a.description,
            "icon": a.icon,
            "earned": a.earned,
        }
        for a in achievements
    ]


init_achievements_db()