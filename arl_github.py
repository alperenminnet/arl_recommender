import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
# Gerekli import işlemlerini yaptık.

df_ = pd.read_excel(r"C:\Users\ALPEREN MİNNET\Desktop\DSMLBC\datasets\online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy() # Kopya df oluşturduk.

def outlier_threshold(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3-quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace = True)
    dataframe = df.loc[(df["StockCode"] != "POST")]
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_threshold(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df) # Dataframe ön işleme işlemlerini yaptık.

df = df[df["Country"] == "Germany"] # Sadece Alman müşterileri seçtik.

def create_invoice_product_df(dataframe, id=False):
    if id:

        return dataframe.groupby(["Invoice","StockCode"])["Quantity"].sum(). \
            unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum(). \
            unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

df_invoice_product = create_invoice_product_df(df , id = True) # Ürünlerin StockCode kısmını kullanmak istediğimiz için id=True kullandık.

frequent_itemset=apriori(df_invoice_product, min_support=0.01, use_colnames=True) # Her bir ürünün tek başına ve diğer ürünlerle alınma olasılığını hesapladık.

rules = association_rules(frequent_itemset ,metric="support", min_threshold=0.01) # Ürünlerin birliktelik kurallarını çıkardık

# Kullanıcı 1 aldığı ürün: 21987
# Kullanıcı 2 aldığı ürün: 23235
# Kullanıcı 3 aldığı ürün: 22747
# Bu kullanıcıların sepetlerine ekledikleri ürünlere ürün önerisi yapacağız. Ama ilk önce ürünlerin isimlerini öğrenelim.

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df, 21987) # ['PACK OF 6 SKULL PAPER CUPS']
check_id(df, 23235) # ['STORAGE TIN VINTAGE LEAF']
check_id(df, 22747) # ["POPPY'S PLAYHOUSE BATHROOM"]
# Ürünlerin isimlerini öğrendik.

def arl_recommender(rules_df, product_id, rec_count = 1):

    sorted_rules = rules_df.sort_values("lift", ascending=False)

    recommendation_list = []

    for i,product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.loc[i]["consequents"]))

    recommendation_list = list(dict.fromkeys(item for item_list in recommendation_list for item in item_list))

    return recommendation_list[:rec_count]

# Ürün önermek için fonksiyonu yazdık.

arl_recommender(rules, 21987, 1) # Önerilen ürün [21988]
arl_recommender(rules, 23235, 2) # Önerilen ürünler [23244, 23236]
arl_recommender(rules, 22747, 3) # Önerilen ürünler [22745, 22746, 22748]
# Ürünleri önerdik.

#Ürün isimleri:
check_id(df, 21988) # 1.Müşteriye önerilen ürün:['PACK OF 6 SKULL PAPER PLATES']

check_id(df, 23244)
check_id(df, 23236) # 2.Müşteriye önerilen ürünler:['ROUND STORAGE TIN VINTAGE LEAF'], ['DOILEY STORAGE TIN']

check_id(df, 22745)
check_id(df, 22746)
check_id(df, 22748) # 3.Müşteriye önerilen ürünler:["POPPY'S PLAYHOUSE BEDROOM "],
# ["POPPY'S PLAYHOUSE LIVINGROOM "],["POPPY'S PLAYHOUSE KITCHEN"]