# Курсовая работа: интеграция BIND9 и Active Directory (Split-DNS)

Тема: «Организация интеграции корпоративного BIND9 и доменной зоны Active Directory
через условную пересылку (Split-DNS)».

- `Kursovaya_Split-DNS_BIND9_AD.docx` — готовая работа (ГОСТ 7.32-2017, 30 с., основной текст ~22 с.).
- `build.js` — генератор документа (весь текст работы находится здесь).
- `update_toc.py` — заполняет «Содержание» номерами страниц через LibreOffice.

Пересборка:

```bash
node build.js && python3 update_toc.py Kursovaya_Split-DNS_BIND9_AD.docx
```

Перед сдачей заполните титульный лист (поля в квадратных скобках).
