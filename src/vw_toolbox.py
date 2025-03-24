from talib import EMA, WILLR, MFI, LINEARREG_SLOPE, ROCR

def ask_user_input_int(msg: str, error_handling: bool = True) -> int:
    """
    Prompt the user to input an integer.

    Parameters:
    -----------
    msg : str
        The message to display to the user.
    error_handling : bool, optional
        Flag to indicate if error handling should be used (default is True).

    Returns:
    --------
    int
        The integer input by the user.
    """
    if error_handling:
        try:
            return int(input(msg).strip())
        except ValueError:
            print("Invalid input")
            return ask_user_input_int(msg, error_handling)
    else:
        return int(input(msg))

def ask_user_input_num(msg: str, error_handling: bool = True) -> float:
    """
    Prompt the user to input a floating number.

    Parameters:
    -----------
    msg : str
        The message to display to the user.
    error_handling : bool, optional
        Flag to indicate if error handling should be used (default is True).

    Returns:
    --------
    float
        The floating number input by the user.
    """
    if error_handling:
        try:
            return float(input(msg).strip())
        except ValueError:
            print("Invalid input")
            return ask_user_input_num(msg, error_handling)
    else:
        return float(input(msg))

def convert_list_to_a_string(contents_to_print: list[str], delimiter: str = " ") -> str:
    """
    Return the contents of a list in one line separated by a delimiter.

    Args:
        contents_to_print (list[str]): The list of strings to print.
        delimiter (str): The delimiter to separate the strings. Default is a space.
    """
    return (delimiter.join(contents_to_print))

def print_in_one_line(contents_to_print: list[str], delimiter: str = " ") -> None:
    """
    Print the contents of a list in one line separated by a delimiter.

    Args:
        contents_to_print (list[str]): The list of strings to print.
        delimiter (str): The delimiter to separate the strings. Default is a space.
    """
    print(convert_list_to_a_string(contents_to_print, delimiter))


def test() -> None:
    """Test function for the tool."""
    #print_with_delimiter("Enter an integer", "*")
    contacts = []
    contacts.append("Kim")
    contacts.append("John")
    contacts.append("Denise")
    print_in_one_line(contacts)
    print_in_one_line(contacts, ">.")
    line_of_contacts = convert_list_to_a_string(contacts)
    print(f"Line of contacts: {line_of_contacts}")
    #print(contacts)
    #print(contacts, sep=", ")
    #print(contacts, end=" ")
    #print(" ".join(contacts))
    #values = [1, 2, 3, 4, 5]
    #values = [1, 2, 3, 4, 5]
    #print(' '.join(map(str, values)))

    #for value in values:
    #    print(value, end='@')

import os
def ask_valid_directory_path(msg: str, error_handling: bool = True) -> str:

    if error_handling:
        try:
            path = input(msg).strip()
            path = path.replace("/", os.sep)
            path = path.replace("\\", os.sep)
            if os.name == "nt":  # Check if the platform is Windows
                path = f"{path.replace(os.sep, "\\\\")}{os.sep}{os.sep}"
            else:
                path = f"{path}{os.sep}{os.sep}"
            # Check if the directory exists
            if os.path.isdir(path):
                return path
            else: # display error message and call the function again
                print(f"Invalid directory path: {path}")
                return ask_valid_directory_path(msg, error_handling)
        except ValueError:
            print("Invalid input")
            return ask_valid_directory_path(msg, error_handling)
    else:
        return input(msg).strip()

import pandas as pd
import yfinance as yf
def get_hist_prices(ticker: str, price_period: str) -> pd.DataFrame:
    """
    Get historical stock prices for a given ticker and date range.

    Args:
        ticker (str): The stock ticker.
        price_period (str): The price period, e.g. "1d", "1mo", "1y".

    Returns:
        pd.DataFrame: The historical stock prices.
    """
    hist_price_df = yf.download(  # or pdr.get_data_yahoo(...
            # tickers list or string as well
            tickers = ticker,

            # use "period" instead of start/end
            # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            # (optional, default is '1mo')
            period = price_period,

            # fetch data by interval (including intraday if period < 60 days)
            # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            # (optional, default is '1d')
            # 1m limitation: only 7 days
            interval = "1d",

            # group by ticker (to access via data['SPY'])
            # (optional, default is 'column')
            group_by = 'ticker',
            #group_by = 'column',

            # adjust all OHLC automatically
            # (optional, default is False)
            # False is the default for browser download from Yahoo Finance website
            auto_adjust = True,

            # download pre/post regular market hours data
            # (optional, default is False)
            prepost = True,

            # use threads for mass downloading? (True/False/Integer)
            # (optional, default is True)
            threads = True,

            # proxy URL scheme use use when downloading?
            # (optional, default is None)
            proxy = None
        )
    return hist_price_df

import math

def calculate_log_return(open_price, close_price):
    """
    Calculate the logarithmic return between open and close prices.
    
    Parameters:
    open_price (float): The opening price
    close_price (float): The closing price
    
    Returns:
    float: The logarithmic return
    """
    if open_price <= 0 or close_price <= 0:
        raise ValueError("Prices must be positive values")
    
    return math.log(close_price / open_price)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_price_changes(df, price_change_col, title, xlabel, image_path):
    """
    Analyze and visualize price change distribution, dividing it into quintiles.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the price change data
    price_change_col (str): Name of the column containing price change ratios
    
    Returns:
    pandas.DataFrame: DataFrame with quintile statistics
    """
    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()
    
    # Calculate quintiles (5 equal parts of 20%)
    quintiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    quintile_values = data[price_change_col].quantile(quintiles).values
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Distribution with KDE and quintile boundaries
    plt.subplot(2, 1, 1)
    sns.histplot(data[price_change_col], kde=True, color='skyblue')
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Add vertical lines for quintile boundaries
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for i, quint_val in enumerate(quintile_values[1:-1], 1):
        plt.axvline(quint_val, color=colors[i], linestyle='--', 
                   label=f'Q{i} ({quintiles[i]*100:.0f}%): {quint_val:.4f}')
    
    # Add line for no change
    #plt.axvline(1.0, color='black', linestyle='-', label='No Change (1.0)')
    plt.legend(fontsize=10)
    
    # Plot 2: Box plot with quintile ranges
    plt.subplot(2, 1, 2)
    
    # Create labels for the quintiles
    quintile_labels = []
    quintile_data = []
    quintile_stats = []
    
    # Create a DataFrame to store quintile statistics
    stats_df = pd.DataFrame(columns=['Quintile', 'Range', 'Min', 'Max', 'Mean', 'Std Dev'])
    
    # Process each quintile
    for i in range(len(quintiles)-1):
        if i == 0:
            mask = (data[price_change_col] >= quintile_values[i]) & (data[price_change_col] <= quintile_values[i+1])
            label = f"Q{i+1}: Bottom 20%"
        elif i == 2:  # Middle quintile (the median part)
            mask = (data[price_change_col] > quintile_values[i]) & (data[price_change_col] <= quintile_values[i+1])
            label = f"Q{i+1}: Middle 20%"
        elif i == 4:
            mask = (data[price_change_col] > quintile_values[i]) & (data[price_change_col] <= quintile_values[i+1])
            label = f"Q{i+1}: Top 20%"
        else:
            mask = (data[price_change_col] > quintile_values[i]) & (data[price_change_col] <= quintile_values[i+1])
            label = f"Q{i+1}: {i*20+1}-{(i+1)*20}%"
            
        quintile_data.append(data.loc[mask, price_change_col])
        quintile_labels.append(label)
        
        # Calculate statistics for this quintile
        quintile_min = data.loc[mask, price_change_col].min()
        quintile_max = data.loc[mask, price_change_col].max()
        quintile_mean = data.loc[mask, price_change_col].mean()
        quintile_std = data.loc[mask, price_change_col].std()
        
        # Add to statistics DataFrame
        stats_df.loc[i] = [
            label, 
            f"{quintile_values[i]:.4f} to {quintile_values[i+1]:.4f}",
            quintile_min,
            quintile_max,
            quintile_mean,
            quintile_std
        ]
    
    # Plot boxplots for each quintile
    plt.boxplot(quintile_data, labels=quintile_labels)
    plt.title(title, fontsize=14)
    plt.ylabel(xlabel, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('price_change_quintiles.png')
    if image_path is not None:
        plt.savefig(image_path)
    else:
        plt.show()
    
    # Print quintile statistics
    print("\nQuintile Statistics:")
    print(stats_df.to_string(index=False))
    
    return stats_df

def calculate_technical_indicators(ticker, df, short_period, long_period):
    df[(ticker, 'mfi')] = MFI(df[(ticker, 'High')], df[(ticker, 'Low')], df[(ticker, 'Close')], df[(ticker, 'Volume')], timeperiod=short_period)
    df[(ticker, 'willr')] = WILLR(df[(ticker, 'High')], df[(ticker, 'Low')], df[(ticker, 'Close')], timeperiod=short_period)
    df[(ticker, 'roc_short')] = ROCR(df[(ticker, 'Close')], timeperiod=short_period)
    df[(ticker, 'roc_intraday')] = (df[(ticker, 'Close')] / df[(ticker, 'Open')])
    df[(ticker, 'price_ema_short')] = df[(ticker, 'Close')] / EMA(df[(ticker, 'Close')], timeperiod=short_period)
    df[(ticker, 'price_ema_long')] = df[(ticker, 'Close')] / EMA(df[(ticker, 'Close')], timeperiod=long_period)
    df[(ticker, 'lr_slope_short')] = LINEARREG_SLOPE(df[(ticker, 'Close')], timeperiod=short_period)
    df[(ticker, 'lr_slope_long')] = LINEARREG_SLOPE(df[(ticker, 'Close')], timeperiod=long_period)
    return df

def add_numeric_target(ticker, df, nof_days, target_suffix):
    df[(ticker, f"NextDay{nof_days}{target_suffix}")] = np.log(df[(ticker, 'Close')].shift(- nof_days) / df[(ticker, 'Close')])
    return df

def add_categorical_target(ticker, df, nof_days, target_suffix):
    mean = df[(ticker, f"NextDay{nof_days}{target_suffix}")].mean()
    std = df[(ticker, f"NextDay{nof_days}{target_suffix}")].std()
    thresholds = [float('-inf'), mean - 1.5*std, mean - 0.5*std, mean + 0.5*std, mean + 1.5*std, float('inf')]
    labels = [-2, -1, 0, 1, 2]
    df[(ticker, f'NextDay{nof_days}{target_suffix}Ctg')] = pd.cut(df[(ticker, f"NextDay{nof_days}{target_suffix}")], bins=thresholds, labels=labels, include_lowest=True)
    return df

def trim_beginning_nan_rows(df, ticker, long_period):
    df = df.iloc[long_period:]
    return df

def drop_columns(df, ticker):
    df.drop(columns=[(ticker, 'Open'), (ticker, 'High'), (ticker, 'Low'), (ticker, 'Close'), (ticker, 'Volume')], inplace=True)
    return df


if __name__ == "__main__":
    #test()
    #print(ask_valid_directory_path("Enter directory: "))
    #aapl_df = get_hist_prices("AAPL", "2y")
    #print(aapl_df.head())
    #print(aapl_df.info())
    #print(f"Log Return: {calculate_log_return(219.5, 220.83)}")
    #print(f"Log Return: {calculate_log_return(219.5, 217.5)}")
    #print(f"Log Return: {calculate_log_return(219.5, 202.5)}")
    data = [1.0232331250638,1.03240401504192,1.03066043202127,1.04279334535727,1.04077450976802,1.03416044638914,1.04267226030744,1.01967907244797,1.01033864171638,1.01516707668533,1.00909291349873,1.00681645063429,1.00580202863581,1.00010307471314,1.00659863622427,1.0197040701119,1.00688194807822,1.0066080102338,1.02194999584634,1.02945557248887,1.02492492068698,1.01406533428716,1.02695481186259,1.01268092944608,1.00970428494228,0.987083468300462,0.989851316547514,0.948160864855364,0.927886011634526,0.928623093158547,0.916143959816864,0.921074289974812,0.909105113462203,0.914748587314775,0.908387173529185,0.918103910927873,0.911413694746175,0.960083645751818,0.984497693745911,0.987037257067879,1.01781578869403,0.992404369675457,1.00461231164737,1.00406774243336,1.03758817429363,1.06275118069672,1.07971277071073,1.08579281818619,1.07882174506423,1.03204892096439,0.980344408210214,1.01020519495368,1.00419897458034,0.978411860229424,0.946176352781029,0.936530986043897,0.931548239966633,0.939353935617477,0.943964149579103,0.959433521403788,0.979556099188835,0.9809742584739,0.981712760404459,0.975382680962478,0.978302105787061,0.971264177309151,0.97828711269442,0.976288195641279,0.962751965748066,0.98957216691645,1.00563462978423,1.01544713095001,1.01652665238372,1.0373923783655,1.0549785471437,1.05870302139191,1.04462355841484,1.02860431191354,1.02755211212354,1.0125530882153,1.00314451453443,0.974026856895577,0.966534391761572,0.972251881456197,0.9516129666519,0.923523851047756,0.940564628296455,0.952831223971982,0.96398544757892,0.989365387404182,1.01202537090192,1.02180684832002,1.03601159815855,1.04831652496282,1.06890701662219,1.09299545650635,1.10953265696002,1.08663728655984,1.09906271326212,1.08212730432361,1.06977504110563,1.07523305444038,1.06958777430301,1.04989069713829,1.04741655601705,1.04281714266502,1.01818668443272,1.03030308819678,1.01029657786954,1.0103185953366,1.00806496833995,0.998629315777872,1.01028994099256,1.0088124704138,1.01547236530284,1.03021516660279,1.01786183684268,1.02263660062796,1.04536105705961,1.04295873962333,1.0330998425388,1.0341023147235,1.01819874127026,1.01305119309219,1.00211043328715,0.989218795430693,0.999327118743597,0.991988007278255,0.977874363537264,0.971833707685475,0.939616220594491,0.940578847023367,0.92368237841896,0.929938733126713,0.953153913090925,0.956301625239612,0.964465154431551,0.960859377688456,0.960429624034167,0.953773625372359,0.984055147002583,1.02377200324954,1.0530481190464,1.070151293571,1.05184309467275,1.05055641683115,1.0428593512005,1.03680151183141,1.03125002322128,1.0240156053619,1.00941543943843,0.990616585221834,0.970192205367801,0.967971502356238,0.969873952583634,0.973830289600107,0.969871881023118,0.982699325482147,0.977357737373388,0.985301525905271,0.999918503026293,0.985200819093383,0.982204107712649,0.968625814212193,0.964356335284232,0.974633139507602,0.970438089598995,0.959279881678565,0.975848377462232,0.980436774826801,0.981536808539847,0.977156583926334,0.960451887607829,0.936990555270901,0.927599713093353,0.916635061761985,0.935404273147616,0.953576910519734,0.948529753099053,0.943280801002902,0.95712301702075,0.960814717194431,0.992118845436335,1.03503410237453,1.05646883579802,1.01402363470949,1.00907871563615,0.98900146590324,0.979680244528781,1.01273884836769,0.991213833840639,0.984996079365473,0.971908700850883,0.963482426782578,0.944870587106702,0.989554686417389,0.977768683212902,0.99309327134596,0.98862757337614,1.00998198443015,1.02956631370629,1.01564425314935,1.00319846509996,0.990274201987542,0.989456031365646,0.972992209640692,0.984505795912632,0.983674242030643,1.00739065606453,0.970578258357405,0.958934977885126,1.00469042519041,1.00560854500956,1.00773796677497,1.03585979220214,1.11139396249297,1.09569470035901,1.09286988334417,1.08117389853474,1.08640893099598,1.08268348561545,1.07511625371187,1.10188586521408,1.12213445499905,1.09863887534752,1.03679526467891,1.05277154402067,1.05598093125942,1.04607045138302,1.01388887273786,1.03785840663851,1.01991624499174,1.01525918486006,1.00827528237729,1.01269492438515,1.02190976531408,1.01732617089516,1.01829989354187,1.01875327564588,1.05356370425084,1.01652799652892,1.09032056709562,1.11971206541308,1.11997489433057,1.10527954200865,1.11668298436499,1.10259836396884,1.07050590696647,1.06689646996067,1.05713855370862,1.08259123633048,1.0294472971347,1.00483409371126,0.983103036702583,1.02004798870935,1.01661517732925,1.03387942286546,1.07945444793122,1.09798054486853,1.09868346915315,1.11436361154552,1.06715123314465,1.0767864067735,1.11290477341063,1.08336795611268,1.03908832959936,1.01187081685367,0.991031222091365,0.983056772667401,0.983951470521581,0.938020361252782,0.955705943546691,0.94543247934993,0.931058037063178,0.931777593843632,0.970290110761902,0.974038710350663,0.98016140588114,0.934407932362237,0.920981221772459,0.960098882870442,0.980780646841218,0.992108737762601,0.997900407080684,1.01245937548063,0.99953458978236,1.03031746526995,1.02934435134868,1.08066828792828,1.09430193846079,1.08026908718854,1.05381787551362,1.05023372922584,1.04436165753199,1.03055087783754,1.02151360546564,1.02256142059549,1.01305016682587,0.986188054009176,0.975012152341756,0.982243827275363,0.983476636751427,0.973858295043415,0.96887930159402,0.976450498532796,0.983575469871279,0.968275332703422,0.944628858135268,0.973156077195413,0.999275507880193,1.02918423745578,1.03342084791708,1.02516855285961,1.03298354459841,1.01666213261754,1.02132241360397,1.02377527511322,1.07710792355087,1.0434522346658,1.02759521647898,0.986018240331519,0.99386509152103,0.978893528302131,0.992963033782778,1.01400362971748,1.00668066675064,0.998946471727572,0.992703897533491,1.03377396222992,1.02204784143081,1.02871451775865,1.03615519842189,1.06671471874668,1.04469149399502,1.00531495530542,1.00668013095196,1.01696327572747,1.00907909376069,0.999230247990804,0.992751793017104,0.97312086454634,0.948553208245204,0.938810853833365,0.947384022797603,0.965158626883349,0.986598398721482,0.981849122091229,0.961768217562628,0.960656909050798,0.979433588108716,1.01133677097489,1.01048643269405,1.02820089552784,1.02273954171474,1.02932802549608,1.00567709485264,1.0128216216289,1.03853185346847,1.04829859640283,1.04357675418791,1.03991760485134,1.06484443872648,1.06416099248166,1.0645260174783,1.06131005256996,1.0626640433764,1.07343282907641,1.06398420110141,1.04862589764097,1.05546337996327,1.04550622701513,1.04778997132524,1.0446321635496,1.0207399079179,1.02777319656053,1.04797404688587,1.03452890712753,1.042095568187,1.05083363043914,1.03077105996257,1.01640271841418,0.997530316254848,0.962008928625235,0.981092538275592,0.9808239326948,0.9517466449459,0.950757949762677,0.917312101678703,0.904949491315715,0.912711728160794,0.943179983897059,0.91150865211769,0.943120711793093,0.91485861973188,0.913591799228441,0.923413547971177,0.91792338148546,0.970487619794038,1.01646756838993,1.02606314909124,0.998822881715282,1.03390873127144,0.991434046060235,1.04563424405365,1.03860078188405,1.04274344399183,1.02177038996302,0.991474365279155,0.977401830477761,0.990685261893686,1.0177009215296,1.03758022099083,1.07336867461872,1.05300352197585,1.05863242588489,1.05402612867054,1.08672706833156,1.08517460974967,1.03327317365502,1.00181537528363,1.00128347435168,0.973139790841059,0.965067253641431,0.962714951582148,0.957287556759585,0.97361028040316,0.920598907764667,0.893944309989127,0.902729219406544,0.883607205980725,0.882773770782245,0.899046343307435]
    # Create DataFrame
    df = pd.DataFrame(data, columns=['price_ratio'])
    stats = analyze_price_changes(df, 'price_ratio')
