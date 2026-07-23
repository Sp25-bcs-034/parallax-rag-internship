import pandas as pd 
from src.clean_data import cleaning 
def test_Cleaning ():
    df = pd.DataFrame ( { "text" :["hello","",None,"hello"]
                         , "label":[1,2,3,1]})
    cleaned = cleaning(df)
    assert cleaned.isnull().sum().sum()==0
    