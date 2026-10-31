"""Telegram notification subscribers."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.database import models as db


class SubscriberRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def subscribe(
        self, chat_id: int, username: str | None = None, competitions: list[str] | None = None
    ) -> db.TelegramSubscriber:
        row = self.session.scalar(
            select(db.TelegramSubscriber).where(db.TelegramSubscriber.chat_id == chat_id)
        )
        if row is None:
            row = db.TelegramSubscriber(chat_id=chat_id)
            self.session.add(row)
        row.username = username
        row.competitions = sorted(competitions or [])
        row.active = True
        row.daily_report = True
        self.session.flush()
        return row

    def unsubscribe(self, chat_id: int) -> bool:
        row = self.session.scalar(
            select(db.TelegramSubscriber).where(db.TelegramSubscriber.chat_id == chat_id)
        )
        if row is None or not row.active:
            return False
        row.active = False
        self.session.flush()
        return True

    def erase(self, chat_id: int) -> bool:
        """Delete every stored personal data of the chat (GDPR right to erasure)."""
        row = self.session.scalar(
            select(db.TelegramSubscriber).where(db.TelegramSubscriber.chat_id == chat_id)
        )
        if row is None:
            return False
        self.session.delete(row)
        self.session.flush()
        return True

    def active(self) -> list[db.TelegramSubscriber]:
        return list(
            self.session.scalars(
                select(db.TelegramSubscriber).where(
                    db.TelegramSubscriber.active.is_(True),
                    db.TelegramSubscriber.daily_report.is_(True),
                )
            )
        )
