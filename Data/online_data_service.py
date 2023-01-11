import pandas as pd
from gspread_dataframe import set_with_dataframe
import gspread

# informations sur le dataset en ligne
GSheetName = "ChatBot_Dataset"
LexiqueGShetName = "Lexique"
TabName = "data"


def write_data_to_sheet(gSheetName, tabName, dataframe):
    gc = gspread.service_account(filename="../Google Sheets/chatbot-ensa-service.json")
    sh = gc.open(gSheetName)
    worksheet = sh.worksheet(tabName)
    set_with_dataframe(worksheet, dataframe)


def get_data_from_sheet(gSheetName, tabName):
    gc = gspread.service_account(filename="../Google Sheets/chatbot-ensa-service.json")
    sh = gc.open(gSheetName)
    worksheet = sh.worksheet(tabName)
    df = pd.DataFrame(worksheet.get_all_records())
    '''
    on va filtrer les datasets qui ont des indices -1 c a d qui n'ont pu etre correctement predit et qui n'ont pas
    encore ete rectifie par une personne
    '''
    if "target" in df.columns:
        df = df[df["target"] != -1]
    return df

