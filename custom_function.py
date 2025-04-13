
country_mapping = {
    1:'Australia',
    2:'Austria',
    3:'Belgium',
    4:'British Virgin Islands',
    5:'Cayman Islands',
    6:'Christmas Island',
    7:'Croatia',
    8:'Cyprus',
    9:'Czech Republic',
    10:'Denmark',
    11:'Estonia',
    12:'unidentified',
    13:'Faroe Islands',
    14:'Finland',
    15:'France',
    16:'Germany',
    17:'Greece',
    18:'Hungary',
    19:'Iceland',
    20:'India',
    21:'Ireland',
    22:'Italy',
    23:'Latvia',
    24:'Lithuania',
    25:'Luxembourg',
    26:'Mexico',
    27:'Netherlands',
    28:'Norway',
    29:'Poland',
    30:'Portugal',
    31:'Romania',
    32:'Russia',
    33:'San Marino',
    34:'Slovakia',
    35:'Slovenia',
    36:'Spain',
    37:'Sweden',
    38:'Switzerland',
    39:'Ukraine',
    40:'United Arab Emirates',
    41:'United Kingdom',
    42:'USA',
    43:'biz (.biz)',
    44:'com (.com)',
    45:'int (.int)',
    46:'net (.net)',
    47:'org (*.org)'
}
page1={
    1:'trousers',
    2:'skirts',
    3:'blouses',
    4:'sale'
}

colors={
    1:'beige',
    2:'black',
    3:'blue',
    4:'brown',
    5:'burgundy',
    6:'gray',
    7:'green',
    8:'navy blue',
    9:'of many colors',
    10:'olive',
    11:'pink',
    12:'red',
    13:'violet',
    14:'white'
}
 
locations={
    1:'top left',
    2:'top in the middle',
    3:'top right',
    4:'bottom left',
    5:'bottom in the middle',
    6:'bottom right'

}

model_photo={
    1:'en face',
    2:'profile'

}

clothing_model={

    'C20':'C', 'B26':'B','C13':'C','B11':'B','B31':'B','C38':'C',
    'C24':'C','C45':'C','B24':'B','A11':'A','P39':'P','P18':'P','P16':'P','P11':'P','A3':'A','P1':'P','A13':'A',
    'C26':'C','B17':'B','A7':'A','C12':'C','A2':'A','P2':'P','P4':'P','C18':'C','P3':'P','P43':'P','C41':'C',
    'C10':'C','C25' :'C','P60' :'P','P77' :'P','C33' :'C','A10' :'A','B34' :'B','P8' :'P','A25':'A','A6' :'A','B10':'B',
    'P12':'P', 'A30':'A', 'C14':'C','C19':'C', 'C40':'C','A8' :'A','A21' :'A','A22' :'A','A5' :'A','C11':'C','A16' :'A',
    'A29' :'A','B20' :'A','C5' :'C','P55' :'P','P80' :'P','P51' :'P','B25' :'B','C35' :'C','C2' :'C','C17' :'C',
    'P14' :'P','P5':'P', 'A39':'A', 'C7':'C', 'P20':'P', 'P67':'P', 'P49':'P', 'P15':'P', 'C44':'C', 'A14':'A', 
    'C9':'C', 'P57':'P', 'P7':'P', 'A1':'A','A38':'A', 'B2':'B', 'P25':'P', 'B27':'B', 'P10':'P', 'P72':'P',
    'B32':'B', 'A33':'A', 'P17':'P', 'C54':'C', 'C56':'C', 'B4':'B','A4':'A', 'C27':'C', 'A15':'A', 'C4':'C',
    'A17':'A', 'A41':'A', 'P62':'P', 'A35':'A', 'P48' :'P','C46':'C', 'C6':'C', 'A18':'A','A37':'A', 'A12':'A',
    'P26':'P', 'P63':'P', 'B14':'B', 'C15':'C', 'P40':'P', 'A36':'A', 'B15':'B', 'P34':'P', 'A42':'A', 'C55':'C',
    'B21':'B', 'P61':'P', 'C8':'C', 'A9':'A', 'P33':'P', 'B8':'B', 'B23':'B', 'B1':'B', 'B13':'B', 'C53':'C', 
    'P29':'P', 'C16':'C', 'B6':'B','P73':'P', 'C50':'C', 'B16':'B', 'A20':'A', 'P42':'P', 'P74':'P', 'P35':'P',
    'A31':'A', 'A26':'A', 'B30':'B', 'P50':'P', 'A28':'A','A32':'A' ,'C59':'C', 'P75':'P', 'P70':'P', 'C48':'C', 
    'P47':'P', 'C58':'C', 'P6':'P', 'C51':'C', 'A27':'A', 'P68':'P','C21':'C', 'P38':'P', 'C32':'C', 'C30':'C', 'P23':'P', 'P9':'P',
    'P19':'P', 'P65':'P', 'C23':'C', 'B29':'B', 'B28':'B', 'B19':'B', 'C34':'C','C49':'C', 'C57':'C', 'P64':'P', 
    'B7':'B', 'C52':'C', 'P44':'P','P71':'P', 'P59':'P', 'A23':'A', 'P82':'P', 'P36':'P', 'B12':'B',
    'B33':'B', 'B9':'B', 'C1':'C', 'P32':'P', 'C42':'C', 'C36':'C', 'P30':'P', 'P37':'P', 'C43':'C', 'C39':'C', 
    'P56':'P', 'B3':'B','A34':'A', 'P76':'P', 'B22':'B', 'A43':'A', 'C3':'C', 'P13':'P', 'B5':'B', 'C28':'C',
    'A40':'A', 'C22':'C', 'C47':'C', 'C29':'C','P24':'P', 'A24':'A', 'P58':'P', 'A19':'A', 'P53':'P', 'C37':'C', 
    'P46':'P', 'P69':'P', 'C31':'C', 'P45':'P', 'P52':'P', 'P78':'P','P21':'P', 'P81':'P', 'P41':'P', 'P66':'P', 
    'P27':'P', 'P31' :'P','P79' :'P','P22':'P', 'P54':'P'
}

onehot_cols = ['country','page1_main_category','colour','location','model_photography','page2_clothing_model']
allowed_countries = ['Poland', 'Czech Republic', 'Lithuania', 'net (.net)', 'com (.com)']

def replace_country_func(df):
    df = df.copy()
    df['country'] = df['country'].map(country_mapping)
    return df

def replace_page1_func(df):
    df = df.copy()
    df['page1_main_category'] = df['page1_main_category'].map(page1)
    return df

def replace_colour_func(df):
    df = df.copy()
    df['colour'] = df['colour'].map(colors)
    return df

def replace_location_func(df):
    df = df.copy()
    df['location'] = df['location'].map(locations)
    return df

def replace_model_photo_func(df):
    df = df.copy()
    df['model_photography'] = df['model_photography'].map(model_photo)
    return df

def replace_clothing_photo_func(df):
    df = df.copy()
    df['page2_clothing_model'] = df['page2_clothing_model'].map(clothing_model)
    return df

def filter_countries_func(df):
    df = df.copy()
    df['country'] = df['country'].apply(
        lambda c: c if c in allowed_countries else 'Others'
    )
    return df
