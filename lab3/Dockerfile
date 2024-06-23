# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем зависимости
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Указываем рабочий каталог
WORKDIR /app

# Копируем все файлы в рабочий каталог
COPY . .

# Запуск приложения
CMD ["python", "app.py"]
