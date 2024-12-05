import pandas as pd
from datetime import datetime

# 读取CSV文件
df = pd.read_csv('./worldcup/worldcup.csv')

# 将 'date' 列转换为 datetime 类型
df['date'] = pd.to_datetime(df['date'])

# 判断每一行是否是周末（星期六或星期天）
df['is_weekend'] = (df['date'].dt.weekday >= 5).astype(int)

# 调整列的顺序
df = df[['date', 'is_weekend', 'OT']]

# 将结果保存回新的CSV文件
df.to_csv('./worldcup/worldcup-weekend.csv', index=False)