# Подготовка
Для установки необходимых пакетов:
```
pip install -r requirements.txt
```
В качестве БД используется PostgreSQL. Для её подготовки:
```
createdb mlops
psql -d mlops < schema.sql
```

# Использование
Укажите имя пользователя и пароль для Вашего клиента PostgreSQL в файле `config.yaml`.

Для запуска:
```
python run.py --mode update
```
