import pymongo
import sys

def test_mongo_connection(uri):
    try:
        # 서버 접속에 실패하면 이 부분에서 예외 발생
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        # server_info()로 실제 DB 정보 호출 -> 실패 시 예외 발생
        info = client.server_info()
        print("MongoDB 연결 성공!")
        print("서버 정보:", info)
    except pymongo.errors.ServerSelectionTimeoutError as e:
        print("MongoDB 연결 실패:", e)
        sys.exit(1)
    except Exception as e:
        print("알 수 없는 오류 발생:", e)
        sys.exit(1)

if __name__ == "__main__":
    # 본인 환경에 맞게 URI를 수정
    mongo_uri = "mongodb://username:password@152.70.233.5:27017"
    test_mongo_connection(mongo_uri)
