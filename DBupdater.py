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

    # 오늘 날짜 기준으로 종목코드들을 읽어와서 stock_data DB의 stock_list 에 저장한다.
    def get_code(self):
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

    # 초기 실행 시 위에서 받아온 stock_list 를 이용해 모든 종목들의 일봉 데이터를 받아 db에 저장한다.
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

    # 초기 실행이 아닌 경우 기존 데이터에 추가한다.
    # 문제점 : 새로 생겨서 기존엔 없던 경우에 신규 상장 뿐 아니라 사명변경, 합병, 물적분할 등이 포함되어 기존 데이터가 수정이 필요할 수 있다.
    def add_day_data(self):
        stock_list = pd.read_sql_table('stock_list', self.engine, columns=['Symbol', 'Name'])
        for i in stock_list.index:
            sym = stock_list.Symbol[i]
            name = stock_list.Name[i]

            # 현재 db의 해당종목 테이블의 마지막 날짜 받기
            # 만약 새로 생긴 종목이라 이전 테이블이 없는 경우가 있으므로 테이블이 있는지 확인
            sql_check = f"SELECT TABLE_NAME FROM information_schema.tables " \
                        f"WHERE table_schema='stock_data' AND TABLE_NAME='{name}' LIMIT 1"

            # self.어쩌구 : 해당 종목 테이블이 있으면 True, 없으면 False
            # 종목이 이미 있다면?
            if self.engine.execute(sql_check).fetchone():
                # 해당 종목의 마지막 Date 를 가져와서 add
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
            # 종목이 새로 생겨서 없다면?
            else:
                df_day_data = fdr.DataReader(sym, '2021')
                # 거래 정지 등에는 O, H, L가 0으로 나오는 문제가 있어서 이를 수정
                if len(df_day_data[df_day_data.Open == 0]) != 0:
                    for idx in df_day_data[df_day_data.Open == 0].index:
                        close = df_day_data.loc[idx]['Close']
                        df_day_data['Open'][idx] = close
                        df_day_data['High'][idx] = close
                        df_day_data['Low'][idx] = close

                df_day_data.to_sql(name, self.engine, if_exists='replace')
                print(f'새로 생긴 종목인 {sym} {name}의 일봉 데이터 추가 완료....{i} / {len(stock_list)}')

            """
            새로 생긴 종목으로 취급되는 것들 (2022-04-06 기준, 2022-03-11 실행 후 실행했을 때)
            플래스크 : 젬벡스지오 -> 사명변경
            파이버프로 : 스팩 합병 상장
            크레버스 : 청담러닝의 씨엠에스에듀 흡수합병
            에스에이치엔엘 : 아래스 -> 사명변경, 상폐위기종목
            다올투자증권 : KTB투자증권 -> 사명변경
            네오이뮨텍 : 원인불명
            posco홀딩스 : 물적분할(포스코)
            
            지투파워, 유일로보틱스, 세아메카닉스, 공구우먼 : 신규상장
            
            """


if __name__ == "__main__":
    dbu = DBupdater()
    # stock_list table 초기화 및 실행 당시의 종목 리스트 구성
    dbu.get_code()
    dbu.get_day_data_initial()
    dbu.add_day_data()
