import sqlite3
import datetime

# 初始化数据库和表
def initialize_database(db_name='predictions.db'):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    # 创建物理模型参数表
    c.execute('''
        CREATE TABLE IF NOT EXISTS physical_model_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            param1 REAL,
            param2 REAL,
            param3 REAL
            -- 添加更多参数列
        )
    ''')
    
    # 创建机器学习参数表
    c.execute('''
        CREATE TABLE IF NOT EXISTS ml_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learning_rate REAL,
            n_estimators INTEGER
            -- 添加更多参数列
        )
    ''')
    
    # 创建预测结果表
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            physical_model_params_id INTEGER,
            ml_params_id INTEGER,
            prediction TEXT,
            FOREIGN KEY (physical_model_params_id) REFERENCES physical_model_params(id),
            FOREIGN KEY (ml_params_id) REFERENCES ml_params(id)
        )
    ''')
    
    conn.commit()
    conn.close()

# 添加物理模型参数记录
def add_physical_model_params(conn, param1, param2, param3):
    c = conn.cursor()
    c.execute('''
        INSERT INTO physical_model_params (param1, param2, param3)
        VALUES (?, ?, ?)
    ''', (param1, param2, param3))
    conn.commit()
    return c.lastrowid

# 添加机器学习参数记录
def add_ml_params(conn, learning_rate, n_estimators):
    c = conn.cursor()
    c.execute('''
        INSERT INTO ml_params (learning_rate, n_estimators)
        VALUES (?, ?)
    ''', (learning_rate, n_estimators))
    conn.commit()
    return c.lastrowid

# 添加预测结果记录
def add_prediction(conn, physical_model_params_id, ml_params_id, prediction):
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (timestamp, physical_model_params_id, ml_params_id, prediction)
        VALUES (?, ?, ?, ?)
    ''', (
        datetime.datetime.now().isoformat(),
        physical_model_params_id,
        ml_params_id,
        str(prediction)
    ))
    conn.commit()

# 查询所有记录
def fetch_all_records(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''
        SELECT predictions.timestamp, physical_model_params.param1, physical_model_params.param2, physical_model_params.param3,
               ml_params.learning_rate, ml_params.n_estimators, predictions.prediction
        FROM predictions
        JOIN physical_model_params ON predictions.physical_model_params_id = physical_model_params.id
        JOIN ml_params ON predictions.ml_params_id = ml_params.id
    ''')
    records = c.fetchall()
    conn.close()
    return records

# 示例使用
if __name__ == '__main__':
    db_name = 'predictions.db'
    
    # 初始化数据库和表
    initialize_database(db_name)
    
    # 假设物理模型参数和机器学习参数
    param1 = 1.0
    param2 = 2.0
    param3 = 3.0  # 新的参数
    learning_rate = 0.01
    n_estimators = 100
    prediction = [0.1, 0.9]
    
    # 连接数据库
    conn = sqlite3.connect(db_name)
    
    # 添加物理模型参数和机器学习参数记录
    physical_model_params_id = add_physical_model_params(conn, param1, param2, param3)
    ml_params_id = add_ml_params(conn, learning_rate, n_estimators)
    
    # 添加预测记录
    add_prediction(conn, physical_model_params_id, ml_params_id, prediction)
    
    # 查询所有记录
    records = fetch_all_records(db_name)
    for record in records:
        print(record)
    
    # 关闭数据库连接
    conn.close()
