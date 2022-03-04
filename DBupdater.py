"""
특정 기간의 일봉 데이터를 받아 DB에 저장한다
Input = 시작날짜, 종료날짜
"""
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

class DBupdater:
    def __init__(self):
        """생성자 : DB 연결 및 현재 종목코드 딕셔너리 생성
        업데이트 날짜 확인해서 오늘이 아니면 DB 초기화
        오늘이면 프로그램 종료
        """
        self.engine = create_engine("mysql+pymysql://root:as6114@localhost:3306/stock_data", encoding='utf-8')

    def __del__(self):
        """소멸자 : DB 연결 해제"""

    def get_code(self):
        """종목코드 읽어와서 업데이트 날짜와 함께 DB에 저장"""
        code_list = fdr.StockListing('KRX')[['Symbol', 'Market', 'Name']]
        """
        제거할 코드들
        KONEX 종목 -> 'Market' == 'KONEX'
        7자리 이상 코드 -> 1WR 이런거 붙는 이상한 애들
        앞 두자리가 41~70인 애들 : ETN, 콜/풋옵션
        ETF는 구분이 어려워서 그냥 포함
        """
        code_list = code_list[code_list['Market'] != 'KONEX']
        code_list = code_list[code_list.Symbol.str.len() <= 6]
        # 앞 두자리 41~70인 애들 제거하기 위해서 mask 를 만들자
        for i in range(41, 71):
            i = str(i)
            mask = code_list.Symbol.str.startswith(i)
            code_list = code_list[~mask]
        code_list = code_list[~code_list.Name.str.contains('스팩')]

        # 종목명 대문자, %는 문제를 일으키므로 변경하자
        for i in code_list.index:
            code_list.Name[i] = code_list.Name[i].lower()

        code_list.to_sql('stock_list', self.engine, if_exists='replace', index=False)
        print("stock_list update 완료")

    def get_day_data(self):
        """받아온 종목코드로 FDR을 이용하여 일봉 데이터를 받아 db에 저장"""
        sql = 'SELECT Symbol, NAME FROM stock_list'
        stock_list = pd.read_sql_table('stock_list', self.engine)
        stock_list.drop('Market', axis=1, inplace=True)

        for i in stock_list.index:
            code, name = stock_list.Symbol[i], stock_list.Name[i]
            df_day_data = fdr.DataReader(code, '2020')
            df_day_data.to_sql(name, self.engine, if_exists='replace')
            print(f'{code} {name}의 일봉 데이터 업데이트 완료....{i} / {len(stock_list)}')


if __name__ == "__main__":
    dbu = DBupdater()
    dbu.get_code()
    dbu.get_day_data()