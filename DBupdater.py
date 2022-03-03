"""
특정 기간의 일봉 데이터를 받아 DB에 저장한다
Input = 시작날짜, 종료날짜
"""
import FinanceDataReader as fdr
import pandas as pd
import pymysql
import numpy as np
from sqlalchemy import create_engine

class DBupdater:
    def __init__(self):
        """생성자 : DB 연결 및 현재 종목코드 딕셔너리 생성
        업데이트 날짜 확인해서 오늘이 아니면 DB 초기화
        오늘이면 프로그램 종료
        """
        conn = pymysql.connect(host='localhost', port=3306, db='day_data', user='root', passwd='as6114')
        cursor = conn.cursor()


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

        engine = create_engine('mysql://root:')

        code_list.to_sql('stock_list', )




    def get_day_data(self):
        """받아온 종목코드로 FDR을 이용하여 일봉 데이터를 받아옴"""

    def replace_into_db(self):
        """받아온 일봉 데이터를 DB에 저장"""

if __name__ == "__main__":
    dbu = DBupdater()
