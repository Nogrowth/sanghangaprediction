"""
특정 기간의 일봉 데이터를 받아 DB에 저장한다
Input = 시작날짜, 종료날짜
"""
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import datetime


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
        """종목코드 읽어와서 업데이트 날짜와 함께 stock_data DB의 stock_list 에 저장"""
        # stock_list table 초기화
        sql = 'DROP TABLE if EXISTS `stock_list`'

        # 새로 stock_list 구성
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

        # ETF fund 관련 종목 삭제
        etf_name_list = ['arirang', 'focus', 'hanaro', 'kbstar', 'kindex', 'kodex', 'kosef', 'master', 'sol',
                         'tiger', 'timefolio', 'trex', '마이티', '미래에셋글로벌리츠']
        for etf in etf_name_list:
            code_list = code_list[~code_list.Name.str.startswith(etf)]

        # 종목명에 200 들어가는 미처 안지워진 종목 삭제
        code_list = code_list[~code_list.Name.str.contains('200')]

        code_list.to_sql('stock_list', self.engine, if_exists='replace', index=False)
        print("stock_list update 완료")

    def get_day_data_initial(self):
        """초기 실행 시 받아온 종목코드로 FDR 을 이용하여 일봉 데이터를 받아 db에 저장"""
        # 초기 실행 확인
        sql = "SELECT TABLE_NAME FROM information_schema.tables WHERE table_schema = 'stock_data' AND TABLE_NAME != 'stock_list'"
        check_initial = self.engine.execute(sql).fetchone()
        if type(check_initial) != type(None):
            return None

        stock_list = pd.read_sql_table('stock_list', self.engine, columns=['Symbol', 'Name'])
        for i in stock_list.index:
            code, name = stock_list.Symbol[i], stock_list.Name[i]
            df_day_data = fdr.DataReader(code, '2021')
            # 거래 정지 등에는 O, H, L가 0으로 나오는 문제가 있어서 이를 수정
            if len(df_day_data[df_day_data.Open == 0]) != 0:
                for idx in df_day_data[df_day_data.Open == 0].index:
                    close = df_day_data.loc[idx]['Close']
                    df_day_data['Open'][idx] = close
                    df_day_data['High'][idx] = close
                    df_day_data['Low'][idx] = close

            df_day_data.to_sql(name, self.engine, if_exists='replace')
            print(f'{code} {name}의 일봉 데이터 초기화 완료....{i} / {len(stock_list)}')

    def add_day_data(self):
        """
        기존 stock_data DB에 지난 실행 날짜 다음날부터 오늘까지의 ohlcv 데이터를 추가한다.
        """

        stock_list = pd.read_sql_table('stock_list', self.engine, columns=['Symbol', 'Name'])
        for i in stock_list.index:
            sym = stock_list.Symbol[i]
            name = stock_list.Name[i]

            # 현재 db의 해당종목 테이블의 마지막 날짜 받기
            # 만약 새로 생긴 종목이라 이전 테이블이 없는 경우가 있으므로 테이블이 있는지 확인
            sql_check = f"SELECT TABLE_NAME FROM information_schema.tables " \
                        f"WHERE table_schema='stock_data' AND TABLE_NAME='{name}' LIMIT 1"
            if not self.engine.execute(sql_check).fetchone():
                sql = f'SELECT `Date` FROM `{name}` ORDER BY `Date` DESC LIMIT 1'
                last_date = self.engine.execute(sql).fetchone()[0]
                df_day_data = fdr.DataReader(sym, start=last_date + datetime.timedelta(days=1))
                if len(df_day_data[df_day_data.Open == 0]) != 0:
                    for idx in df_day_data[df_day_data.Open == 0].index:
                        close = df_day_data.loc[idx]['Close']
                        df_day_data['Open'][idx] = close
                        df_day_data['High'][idx] = close
                        df_day_data['Low'][idx] = close
                df_day_data.to_sql(name, self.engine, if_exists='append')
                print(f'{sym} {name}의 일봉 데이터 업데이트 완료....{i} / {len(stock_list)}')


if __name__ == "__main__":
    dbu = DBupdater()
    # stock_list table 초기화 및 실행 당시의 종목 리스트 구성
    # dbu.get_code()
    dbu.get_day_data_initial()
    dbu.add_day_data()
