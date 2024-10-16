# NBA-Defense-Model
Exploring NBA defensive stats to determine the role defense plays in winning.


# Predicting Game Outcome Using Team & Player Stats

NBA Seasons 2019-2022


```python
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
```

    C:\Users\theri\anaconda3\lib\site-packages\pandas\core\computation\expressions.py:21: UserWarning: Pandas requires version '2.8.0' or newer of 'numexpr' (version '2.7.3' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED
    C:\Users\theri\anaconda3\lib\site-packages\pandas\core\arrays\masked.py:62: UserWarning: Pandas requires version '1.3.4' or newer of 'bottleneck' (version '1.3.2' currently installed).
      from pandas.core import (
    C:\Users\theri\anaconda3\lib\site-packages\scipy\__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.26.3
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
    

## 1. Data Cleaning & Processing


```python
# read data source
df_full = pd.read_csv('data_nodupl.csv',index_col=False)
```

    C:\Users\theri\AppData\Local\Temp/ipykernel_7884/2701628176.py:2: DtypeWarning: Columns (68,69,70) have mixed types. Specify dtype option on import or set low_memory=False.
      df_full = pd.read_csv('data_nodupl.csv',index_col=False)
    


```python
df_full.shape
```




    (115715, 83)




```python
#convert game date to datetime field
df_full['game_date'] = pd.to_datetime(df_full['game_date'])
```


```python
# Function to count the number of inactive player names in a comma delimited string
count_names = lambda text: len(str(text).split(',')) if pd.notna(text) else 0

# Apply the function to the 'Inactives' column (replaces list of names with count of names)
df_full['Inactives'] = df_full['Inactives'].apply(count_names)
```


```python
#extract minutes from 'mp' (minutes played) field and convert to integer
get_minutes = lambda text: int(str(text).split(':')[0]) if pd.notna(text) else 0

df_full['mp'] = df_full['mp'].apply(get_minutes)
```


```python
#convert sports betting rates to floats
dkp_cvt = lambda num: float(num) if pd.notna(num) and num != "#NAME?" else 0
fdp_cvt = lambda num: float(num) if pd.notna(num) and num != "#NAME?" else 0
sdp_cvt = lambda num: float(num) if pd.notna(num) and num != "#NAME?" else 0

df_full['DKP_per_minute'] = df_full['DKP_per_minute'].apply(dkp_cvt)
df_full['FDP_per_minute'] = df_full['FDP_per_minute'].apply(fdp_cvt)
df_full['SDP_per_minute'] = df_full['SDP_per_minute'].apply(sdp_cvt)
```


```python
#drop 'duplicate' field
df_full.drop(columns = ['duplicate'],inplace=True)
```


```python
#get list of columns that contain NaN values
columns_with_nan = df_full.columns[df_full.isna().any()].tolist()

print("Columns with NaN values:", columns_with_nan)
```

    Columns with NaN values: ['pf_per_minute', 'last_60_minutes_per_game_starting', 'last_60_minutes_per_game_bench', 'PG%', 'SG%', 'SF%', 'PF%', 'C%', 'active_position_minutes']
    


```python
#drop rows with nan values
df_full_cleaned = df_full.fillna(0)
```


```python
df_full_cleaned.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 115715 entries, 0 to 115714
    Data columns (total 82 columns):
     #   Column                             Non-Null Count   Dtype         
    ---  ------                             --------------   -----         
     0   game_id                            115715 non-null  object        
     1   game_date                          115715 non-null  datetime64[ns]
     2   OT                                 115715 non-null  int64         
     3   H_A                                115715 non-null  object        
     4   Team_Abbrev                        115715 non-null  object        
     5   Team_Score                         115715 non-null  int64         
     6   Team_pace                          115715 non-null  float64       
     7   Team_efg_pct                       115715 non-null  float64       
     8   Team_tov_pct                       115715 non-null  float64       
     9   Team_orb_pct                       115715 non-null  float64       
     10  Team_ft_rate                       115715 non-null  float64       
     11  Team_off_rtg                       115715 non-null  float64       
     12  Inactives                          115715 non-null  int64         
     13  Opponent_Abbrev                    115715 non-null  object        
     14  Opponent_Score                     115715 non-null  int64         
     15  Opponent_pace                      115715 non-null  float64       
     16  Opponent_efg_pct                   115715 non-null  float64       
     17  Opponent_tov_pct                   115715 non-null  float64       
     18  Opponent_orb_pct                   115715 non-null  float64       
     19  Opponent_ft_rate                   115715 non-null  float64       
     20  Opponent_off_rtg                   115715 non-null  float64       
     21  player                             115715 non-null  object        
     22  player_id                          115715 non-null  object        
     23  starter                            115715 non-null  int64         
     24  mp                                 115715 non-null  int64         
     25  fg                                 115715 non-null  int64         
     26  fga                                115715 non-null  int64         
     27  fg_pct                             115715 non-null  float64       
     28  fg3                                115715 non-null  int64         
     29  fg3a                               115715 non-null  int64         
     30  fg3_pct                            115715 non-null  float64       
     31  ft                                 115715 non-null  int64         
     32  fta                                115715 non-null  int64         
     33  ft_pct                             115715 non-null  float64       
     34  orb                                115715 non-null  int64         
     35  drb                                115715 non-null  int64         
     36  trb                                115715 non-null  int64         
     37  ast                                115715 non-null  int64         
     38  stl                                115715 non-null  int64         
     39  blk                                115715 non-null  int64         
     40  tov                                115715 non-null  int64         
     41  pf                                 115715 non-null  int64         
     42  pts                                115715 non-null  int64         
     43  plus_minus                         115715 non-null  int64         
     44  did_not_play                       115715 non-null  int64         
     45  is_inactive                        115715 non-null  int64         
     46  ts_pct                             115715 non-null  float64       
     47  efg_pct                            115715 non-null  float64       
     48  fg3a_per_fga_pct                   115715 non-null  float64       
     49  fta_per_fga_pct                    115715 non-null  float64       
     50  orb_pct                            115715 non-null  float64       
     51  drb_pct                            115715 non-null  float64       
     52  trb_pct                            115715 non-null  float64       
     53  ast_pct                            115715 non-null  float64       
     54  stl_pct                            115715 non-null  float64       
     55  blk_pct                            115715 non-null  float64       
     56  tov_pct                            115715 non-null  float64       
     57  usg_pct                            115715 non-null  float64       
     58  off_rtg                            115715 non-null  int64         
     59  def_rtg                            115715 non-null  int64         
     60  bpm                                115715 non-null  float64       
     61  season                             115715 non-null  int64         
     62  minutes                            115715 non-null  float64       
     63  double_double                      115715 non-null  int64         
     64  triple_double                      115715 non-null  int64         
     65  DKP                                115715 non-null  float64       
     66  FDP                                115715 non-null  float64       
     67  SDP                                115715 non-null  float64       
     68  DKP_per_minute                     115715 non-null  float64       
     69  FDP_per_minute                     115715 non-null  float64       
     70  SDP_per_minute                     115715 non-null  float64       
     71  pf_per_minute                      115715 non-null  float64       
     72  ts                                 115715 non-null  float64       
     73  last_60_minutes_per_game_starting  115715 non-null  float64       
     74  last_60_minutes_per_game_bench     115715 non-null  float64       
     75  PG%                                115715 non-null  float64       
     76  SG%                                115715 non-null  float64       
     77  SF%                                115715 non-null  float64       
     78  PF%                                115715 non-null  float64       
     79  C%                                 115715 non-null  float64       
     80  active_position_minutes            115715 non-null  float64       
     81  outcome                            115715 non-null  int64         
    dtypes: datetime64[ns](1), float64(45), int64(30), object(6)
    memory usage: 72.4+ MB
    


```python
df_full_cleaned.shape
```




    (115715, 82)




```python
df_full_cleaned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_id</th>
      <th>game_date</th>
      <th>OT</th>
      <th>H_A</th>
      <th>Team_Abbrev</th>
      <th>Team_Score</th>
      <th>Team_pace</th>
      <th>Team_efg_pct</th>
      <th>Team_tov_pct</th>
      <th>Team_orb_pct</th>
      <th>Team_ft_rate</th>
      <th>Team_off_rtg</th>
      <th>Inactives</th>
      <th>Opponent_Abbrev</th>
      <th>Opponent_Score</th>
      <th>Opponent_pace</th>
      <th>Opponent_efg_pct</th>
      <th>Opponent_tov_pct</th>
      <th>Opponent_orb_pct</th>
      <th>Opponent_ft_rate</th>
      <th>Opponent_off_rtg</th>
      <th>player</th>
      <th>player_id</th>
      <th>starter</th>
      <th>mp</th>
      <th>fg</th>
      <th>fga</th>
      <th>fg_pct</th>
      <th>fg3</th>
      <th>fg3a</th>
      <th>fg3_pct</th>
      <th>ft</th>
      <th>fta</th>
      <th>ft_pct</th>
      <th>orb</th>
      <th>drb</th>
      <th>trb</th>
      <th>ast</th>
      <th>stl</th>
      <th>blk</th>
      <th>tov</th>
      <th>pf</th>
      <th>pts</th>
      <th>plus_minus</th>
      <th>did_not_play</th>
      <th>is_inactive</th>
      <th>ts_pct</th>
      <th>efg_pct</th>
      <th>fg3a_per_fga_pct</th>
      <th>fta_per_fga_pct</th>
      <th>orb_pct</th>
      <th>drb_pct</th>
      <th>trb_pct</th>
      <th>ast_pct</th>
      <th>stl_pct</th>
      <th>blk_pct</th>
      <th>tov_pct</th>
      <th>usg_pct</th>
      <th>off_rtg</th>
      <th>def_rtg</th>
      <th>bpm</th>
      <th>season</th>
      <th>minutes</th>
      <th>double_double</th>
      <th>triple_double</th>
      <th>DKP</th>
      <th>FDP</th>
      <th>SDP</th>
      <th>DKP_per_minute</th>
      <th>FDP_per_minute</th>
      <th>SDP_per_minute</th>
      <th>pf_per_minute</th>
      <th>ts</th>
      <th>last_60_minutes_per_game_starting</th>
      <th>last_60_minutes_per_game_bench</th>
      <th>PG%</th>
      <th>SG%</th>
      <th>SF%</th>
      <th>PF%</th>
      <th>C%</th>
      <th>active_position_minutes</th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>202204100BRK</td>
      <td>2022-04-10</td>
      <td>0</td>
      <td>A</td>
      <td>IND</td>
      <td>126</td>
      <td>103.9</td>
      <td>0.543</td>
      <td>5.9</td>
      <td>20.8</td>
      <td>0.125</td>
      <td>121.3</td>
      <td>7</td>
      <td>BRK</td>
      <td>134</td>
      <td>103.9</td>
      <td>0.691</td>
      <td>17.9</td>
      <td>29.6</td>
      <td>0.272</td>
      <td>129.0</td>
      <td>Tyrese Haliburton</td>
      <td>halibty01</td>
      <td>1</td>
      <td>39</td>
      <td>7</td>
      <td>14</td>
      <td>0.500</td>
      <td>2</td>
      <td>5</td>
      <td>0.400</td>
      <td>1</td>
      <td>1</td>
      <td>1.00</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>17</td>
      <td>-9</td>
      <td>0</td>
      <td>0</td>
      <td>0.589</td>
      <td>0.571</td>
      <td>0.357</td>
      <td>0.071</td>
      <td>4.6</td>
      <td>9.0</td>
      <td>6.1</td>
      <td>31.6</td>
      <td>2.3</td>
      <td>0.0</td>
      <td>6.5</td>
      <td>15.7</td>
      <td>137</td>
      <td>132</td>
      <td>1.7</td>
      <td>2022</td>
      <td>39.466667</td>
      <td>1</td>
      <td>0</td>
      <td>43.00</td>
      <td>41.8</td>
      <td>44.50</td>
      <td>1.089527</td>
      <td>1.059122</td>
      <td>1.127534</td>
      <td>0.000000</td>
      <td>14.44</td>
      <td>36.176282</td>
      <td>19.535428</td>
      <td>62.0</td>
      <td>35.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>64.741860</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>202204100BRK</td>
      <td>2022-04-10</td>
      <td>0</td>
      <td>A</td>
      <td>IND</td>
      <td>126</td>
      <td>103.9</td>
      <td>0.543</td>
      <td>5.9</td>
      <td>20.8</td>
      <td>0.125</td>
      <td>121.3</td>
      <td>7</td>
      <td>BRK</td>
      <td>134</td>
      <td>103.9</td>
      <td>0.691</td>
      <td>17.9</td>
      <td>29.6</td>
      <td>0.272</td>
      <td>129.0</td>
      <td>Buddy Hield</td>
      <td>hieldbu01</td>
      <td>1</td>
      <td>35</td>
      <td>8</td>
      <td>23</td>
      <td>0.348</td>
      <td>5</td>
      <td>14</td>
      <td>0.357</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.457</td>
      <td>0.457</td>
      <td>0.609</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>14.9</td>
      <td>5.0</td>
      <td>22.1</td>
      <td>3.9</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>28.0</td>
      <td>94</td>
      <td>128</td>
      <td>-2.3</td>
      <td>2022</td>
      <td>35.883333</td>
      <td>0</td>
      <td>0</td>
      <td>41.25</td>
      <td>40.6</td>
      <td>43.25</td>
      <td>1.149559</td>
      <td>1.131444</td>
      <td>1.205295</td>
      <td>0.083604</td>
      <td>23.00</td>
      <td>35.833974</td>
      <td>19.350580</td>
      <td>4.0</td>
      <td>53.0</td>
      <td>43.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>56.265965</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>202204100BRK</td>
      <td>2022-04-10</td>
      <td>0</td>
      <td>A</td>
      <td>IND</td>
      <td>126</td>
      <td>103.9</td>
      <td>0.543</td>
      <td>5.9</td>
      <td>20.8</td>
      <td>0.125</td>
      <td>121.3</td>
      <td>7</td>
      <td>BRK</td>
      <td>134</td>
      <td>103.9</td>
      <td>0.691</td>
      <td>17.9</td>
      <td>29.6</td>
      <td>0.272</td>
      <td>129.0</td>
      <td>Oshae Brissett</td>
      <td>brissos01</td>
      <td>1</td>
      <td>35</td>
      <td>10</td>
      <td>20</td>
      <td>0.500</td>
      <td>5</td>
      <td>10</td>
      <td>0.500</td>
      <td>3</td>
      <td>4</td>
      <td>0.75</td>
      <td>3</td>
      <td>5</td>
      <td>8</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>28</td>
      <td>-9</td>
      <td>0</td>
      <td>0</td>
      <td>0.643</td>
      <td>0.625</td>
      <td>0.500</td>
      <td>0.200</td>
      <td>7.6</td>
      <td>24.8</td>
      <td>13.4</td>
      <td>12.0</td>
      <td>1.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>24.5</td>
      <td>137</td>
      <td>133</td>
      <td>4.4</td>
      <td>2022</td>
      <td>35.783333</td>
      <td>0</td>
      <td>0</td>
      <td>47.00</td>
      <td>45.1</td>
      <td>48.00</td>
      <td>1.313461</td>
      <td>1.260363</td>
      <td>1.341407</td>
      <td>0.139730</td>
      <td>21.76</td>
      <td>30.988194</td>
      <td>16.733827</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>75.0</td>
      <td>5.0</td>
      <td>44.631041</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>202204100BRK</td>
      <td>2022-04-10</td>
      <td>0</td>
      <td>A</td>
      <td>IND</td>
      <td>126</td>
      <td>103.9</td>
      <td>0.543</td>
      <td>5.9</td>
      <td>20.8</td>
      <td>0.125</td>
      <td>121.3</td>
      <td>7</td>
      <td>BRK</td>
      <td>134</td>
      <td>103.9</td>
      <td>0.691</td>
      <td>17.9</td>
      <td>29.6</td>
      <td>0.272</td>
      <td>129.0</td>
      <td>Isaiah Jackson</td>
      <td>jacksis01</td>
      <td>1</td>
      <td>32</td>
      <td>3</td>
      <td>4</td>
      <td>0.750</td>
      <td>0</td>
      <td>0</td>
      <td>0.000</td>
      <td>1</td>
      <td>2</td>
      <td>0.50</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.717</td>
      <td>0.750</td>
      <td>0.000</td>
      <td>0.500</td>
      <td>0.0</td>
      <td>16.7</td>
      <td>5.6</td>
      <td>0.0</td>
      <td>2.9</td>
      <td>2.5</td>
      <td>29.1</td>
      <td>8.6</td>
      <td>89</td>
      <td>128</td>
      <td>-9.2</td>
      <td>2022</td>
      <td>32.016667</td>
      <td>0</td>
      <td>0</td>
      <td>15.75</td>
      <td>17.6</td>
      <td>17.75</td>
      <td>0.491931</td>
      <td>0.549714</td>
      <td>0.554399</td>
      <td>0.156169</td>
      <td>4.88</td>
      <td>22.514103</td>
      <td>14.625000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>24.0</td>
      <td>76.0</td>
      <td>53.345389</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>202204100BRK</td>
      <td>2022-04-10</td>
      <td>0</td>
      <td>A</td>
      <td>IND</td>
      <td>126</td>
      <td>103.9</td>
      <td>0.543</td>
      <td>5.9</td>
      <td>20.8</td>
      <td>0.125</td>
      <td>121.3</td>
      <td>7</td>
      <td>BRK</td>
      <td>134</td>
      <td>103.9</td>
      <td>0.691</td>
      <td>17.9</td>
      <td>29.6</td>
      <td>0.272</td>
      <td>129.0</td>
      <td>T.J. McConnell</td>
      <td>mccontj01</td>
      <td>1</td>
      <td>30</td>
      <td>5</td>
      <td>15</td>
      <td>0.333</td>
      <td>3</td>
      <td>7</td>
      <td>0.429</td>
      <td>1</td>
      <td>2</td>
      <td>0.50</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>14</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0.441</td>
      <td>0.433</td>
      <td>0.467</td>
      <td>0.133</td>
      <td>0.0</td>
      <td>17.3</td>
      <td>5.8</td>
      <td>19.8</td>
      <td>4.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.7</td>
      <td>104</td>
      <td>126</td>
      <td>-1.7</td>
      <td>2022</td>
      <td>30.866667</td>
      <td>0</td>
      <td>0</td>
      <td>32.75</td>
      <td>34.1</td>
      <td>35.75</td>
      <td>1.061015</td>
      <td>1.104752</td>
      <td>1.158207</td>
      <td>0.097192</td>
      <td>15.88</td>
      <td>25.350000</td>
      <td>15.166667</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>68.220319</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Get sum and mean player stats

'''
df_player_sums = df_full_cleaned.groupby(by = ['season','player','Team_Abbrev'],as_index=False).agg (sumfg = ('fg','sum'),sumfg3=('fg3','sum'),sumft=('ft','sum'),sumorb=('orb','sum'),sumast=('ast','sum'),sumpts=('pts','sum'),sumdrb = ('drb','sum'),sumstl=('stl','sum'),sumblk=('blk','sum'))
df_player_avg = df_full_cleaned.groupby(by = ['season','player','Team_Abbrev'],as_index=False).agg (avgfg = ('fg','mean'),avgfg3=('fg3','mean'),avgft=('ft','mean'),avgorb=('orb','mean'),avgast=('ast','mean'),avgpts=('pts','mean'),avgdrb = ('drb','mean'),avgstl=('stl','mean'),avgblk=('blk','mean'))

df_player_avg = df_player_avg.iloc[:,3:]

df_player_sum_avg = pd.concat([df_player_sums, df_player_avg],axis=1)

df_player_sum_avg.head()
'''
```


```python
df_player_sum_avg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>player</th>
      <th>Team_Abbrev</th>
      <th>sumfg</th>
      <th>sumfg3</th>
      <th>sumft</th>
      <th>sumorb</th>
      <th>sumast</th>
      <th>sumpts</th>
      <th>sumdrb</th>
      <th>sumstl</th>
      <th>sumblk</th>
      <th>avgfg</th>
      <th>avgfg3</th>
      <th>avgft</th>
      <th>avgorb</th>
      <th>avgast</th>
      <th>avgpts</th>
      <th>avgdrb</th>
      <th>avgstl</th>
      <th>avgblk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020</td>
      <td>Aaron Gordon</td>
      <td>ORL</td>
      <td>335</td>
      <td>73</td>
      <td>151</td>
      <td>107</td>
      <td>228</td>
      <td>894</td>
      <td>368</td>
      <td>51</td>
      <td>39</td>
      <td>5.403226</td>
      <td>1.177419</td>
      <td>2.435484</td>
      <td>1.725806</td>
      <td>3.677419</td>
      <td>14.419355</td>
      <td>5.935484</td>
      <td>0.822581</td>
      <td>0.629032</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020</td>
      <td>Aaron Holiday</td>
      <td>IND</td>
      <td>245</td>
      <td>91</td>
      <td>77</td>
      <td>23</td>
      <td>235</td>
      <td>658</td>
      <td>138</td>
      <td>59</td>
      <td>16</td>
      <td>3.181818</td>
      <td>1.181818</td>
      <td>1.000000</td>
      <td>0.298701</td>
      <td>3.051948</td>
      <td>8.545455</td>
      <td>1.792208</td>
      <td>0.766234</td>
      <td>0.207792</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020</td>
      <td>Abdel Nader</td>
      <td>OKC</td>
      <td>248</td>
      <td>98</td>
      <td>104</td>
      <td>30</td>
      <td>76</td>
      <td>698</td>
      <td>176</td>
      <td>48</td>
      <td>44</td>
      <td>1.746479</td>
      <td>0.690141</td>
      <td>0.732394</td>
      <td>0.211268</td>
      <td>0.535211</td>
      <td>4.915493</td>
      <td>1.239437</td>
      <td>0.338028</td>
      <td>0.309859</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020</td>
      <td>Adam Mokoka</td>
      <td>CHI</td>
      <td>12</td>
      <td>6</td>
      <td>2</td>
      <td>7</td>
      <td>4</td>
      <td>32</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>0.111111</td>
      <td>0.388889</td>
      <td>0.222222</td>
      <td>1.777778</td>
      <td>0.166667</td>
      <td>0.222222</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020</td>
      <td>Admiral Schofield</td>
      <td>WAS</td>
      <td>35</td>
      <td>19</td>
      <td>10</td>
      <td>7</td>
      <td>15</td>
      <td>99</td>
      <td>40</td>
      <td>8</td>
      <td>4</td>
      <td>0.795455</td>
      <td>0.431818</td>
      <td>0.227273</td>
      <td>0.159091</td>
      <td>0.340909</td>
      <td>2.250000</td>
      <td>0.909091</td>
      <td>0.181818</td>
      <td>0.090909</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_player_sum_avg.to_excel('nba_player_stats.xlsx',index=False)
```

    C:\Users\theri\AppData\Local\Temp/ipykernel_7884/2179414131.py:1: UserWarning: Pandas requires version '3.0.3' or newer of 'xlsxwriter' (version '3.0.1' currently installed).
      df_player_sum_avg.to_excel('nba_player_stats.xlsx',index=False)
    


```python
#Define the pure defensive and offensive player stats (By pure I mean those stats that can only be regarded as defensive or offensive)--excluding percentages as a fg percentage of 100% is misleading if the player only attempted one fg.
player_def_stats_cols = ['game_id','Team_Abbrev','Opponent_Abbrev','outcome','stl','blk','drb']

player_off_stats_cols = ['game_id','Team_Abbrev','Opponent_Abbrev','outcome','fg','fg3','ft','orb','ast','pts']
```


```python
#create player def stats dataframe
df_player_def = df_full_cleaned[player_def_stats_cols]

#create player off stats dataframe
df_player_off = df_full_cleaned[player_off_stats_cols]

#create df with combined (def+off) player stats
df_player_combined = df_full_cleaned[player_def_stats_cols + player_off_stats_cols]
df_player_combined = df_player_combined.iloc[:,4:]
```


```python
df_player_combined.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stl</th>
      <th>blk</th>
      <th>drb</th>
      <th>game_id</th>
      <th>Team_Abbrev</th>
      <th>Opponent_Abbrev</th>
      <th>outcome</th>
      <th>fg</th>
      <th>fg3</th>
      <th>ft</th>
      <th>orb</th>
      <th>ast</th>
      <th>pts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>202204100BRK</td>
      <td>IND</td>
      <td>BRK</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>10</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>202204100BRK</td>
      <td>IND</td>
      <td>BRK</td>
      <td>0</td>
      <td>8</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>202204100BRK</td>
      <td>IND</td>
      <td>BRK</td>
      <td>0</td>
      <td>10</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>202204100BRK</td>
      <td>IND</td>
      <td>BRK</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>202204100BRK</td>
      <td>IND</td>
      <td>BRK</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>




```python
#aggregate stats by team
df_def_agg = df_player_def.groupby(by=['game_id','Team_Abbrev','Opponent_Abbrev','outcome'],as_index=False).agg (sumdrb = ('drb','sum'),sumstl=('stl','sum'),sumblk=('blk','sum'))
df_off_agg = df_player_off.groupby(by=['game_id','Team_Abbrev','Opponent_Abbrev','outcome'],as_index=False).agg (sumfg = ('fg','sum'),sumfg3=('fg3','sum'),sumft=('ft','sum'),sumorb=('orb','sum'),sumast=('ast','sum'),sumpts=('pts','sum'))
df_combined_agg = df_player_combined.groupby(by=['game_id','Team_Abbrev','Opponent_Abbrev','outcome'],as_index=False).agg (sumfg = ('fg','sum'),sumfg3=('fg3','sum'),sumft=('ft','sum'),sumorb=('orb','sum'),sumast=('ast','sum'),sumpts=('pts','sum'),sumdrb = ('drb','sum'),sumstl=('stl','sum'),sumblk=('blk','sum'))
```


```python
#confirm aggregations are correct

'''
df_player_combined[df_player_combined['game_id'] == '201910240DET']

df_full_cleaned[df_full_cleaned['game_id'] == '201910240DET']
'''
```




    "\ndf_player_combined[df_player_combined['game_id'] == '201910240DET']\n\ndf_full_cleaned[df_full_cleaned['game_id'] == '201910240DET']\n"




```python
#create offensive score feature

# 1. create new column that combines Opponent's Abbrev and the game id
df_combined_agg['opponent_id'] = df_combined_agg.apply(lambda row: row['Opponent_Abbrev'] + '_' + str(row['game_id']),axis = 1)



'''
# 2. create dictionary that maps 'opponent_id' values to teams offensive stats
offense_lead_dict = {}
offense_lead_dict = dict(zip(df_combined_agg['opponent_id'],zip(df_combined_agg['sumfg'],
                                                                df_combined_agg['sumfg3'],df_combined_agg['sumft'],
                                                                df_combined_agg['sumorb'],
                                                                df_combined_agg['sumast'],df_combined_agg['sumpts'])))
'''

# 2. create dictionary that maps 'opponent_id' values to teams offensive stats (removed sumpts)
offense_lead_dict = {}
offense_lead_dict = dict(zip(df_combined_agg['opponent_id'],zip(df_combined_agg['sumfg'],df_combined_agg['sumfg3'],df_combined_agg['sumft'],
                                                               df_combined_agg['sumorb'],df_combined_agg['sumast'])))


# 3. compare team and opponent defensive stats, if two or more team def stats are greater than opponent's def stats then 'good D' otherwise 'bad D'

final_score = [] #stores sum of def_score for each team
for i in range(0,df_combined_agg.shape[0]):
    
    off_score = [] #empty list to store 1s if current teams defensive stat exceeds opponent's
    
    '''
    team_off_tup = (df_combined_agg['sumfg'][i],df_combined_agg['sumfg3'][i],df_combined_agg['sumft'][i],
                   df_combined_agg['sumorb'][i],df_combined_agg['sumast'][i],df_combined_agg['sumpts'][i]) # current team's defensive stats
    
    '''
    team_off_tup = (df_combined_agg['sumfg'][i],df_combined_agg['sumfg3'][i],df_combined_agg['sumft'][i],
                   df_combined_agg['sumorb'][i],df_combined_agg['sumast'][i]) # current team's defensive stats
    
    tupKey_off = df_combined_agg['Team_Abbrev'][i] + '_' + df_combined_agg['game_id'][i] #dictionary key to get opponents defensive stats
    opp_tup = offense_lead_dict[tupKey_off] #opponents defensive stats
    
    #logic to calculate team's defensive score
    if team_off_tup[0] > opp_tup[0]:
        off_score.append(1)
    else:
        off_score.append(0)
    if team_off_tup[1] > opp_tup[1]:
        off_score.append(1)
    else:
        off_score.append(0)
    if team_off_tup[2] > opp_tup[2]:
        off_score.append(1)
    else:
        off_score.append(0)
    if team_off_tup[3] > opp_tup[3]:
        off_score.append(1)
    else:
        off_score.append(0)
    if team_off_tup[4] > opp_tup[4]:
        off_score.append(1)
    else:
        off_score.append(0)
    
    '''
    if team_off_tup[5] > opp_tup[5]:
        off_score.append(1)
    else:
        off_score.append(0)
    '''
        
    #final_score.append(1 if sum(off_score) >= 4 else 0) #sum values in def_score and append to final_score
    final_score.append(1 if sum(off_score) >= 3 else 0) #sum values in def_score and append to final_score

    
df_combined_agg['Offensive Effort'] = final_score
```


```python
#create defensive score feature

# 1. create new column that combines Opponent's Abbrev and the game id
df_combined_agg['opponent_id'] = df_combined_agg.apply(lambda row: row['Opponent_Abbrev'] + '_' + str(row['game_id']),axis = 1)

# 2. create dictionary that maps 'opponent_id' values to teams defensive stats
defense_lead_dict = {}
defense_lead_dict = dict(zip(df_combined_agg['opponent_id'],zip(df_combined_agg['sumdrb'],df_combined_agg['sumstl'],df_combined_agg['sumblk'])))

# 3. compare team and opponent defensive stats, if two or more team def stats are greater than opponent's def stats then 'good D' otherwise 'bad D'

final_score = [] #stores sum of def_score for each team
for i in range(0,df_combined_agg.shape[0]):
    
    def_score = [] #empty list to store 1s if current teams defensive stat exceeds opponent's
    team_tup = (df_combined_agg['sumdrb'][i],df_combined_agg['sumstl'][i],df_combined_agg['sumblk'][i]) # current team's defensive stats
    
    tupKey = df_combined_agg['Team_Abbrev'][i] + '_' + df_combined_agg['game_id'][i] #dictionary key to get opponents defensive stats
    opp_tup = defense_lead_dict[tupKey] #opponents defensive stats
    
    #logic to calculate team's defensive score
    if team_tup[0] > opp_tup[0]:
        def_score.append(1)
    else:
        def_score.append(0)
    if team_tup[1] > opp_tup[1]:
        def_score.append(1)
    else:
        def_score.append(0)
    if team_tup[2] > opp_tup[2]:
        def_score.append(1)
    else:
        def_score.append(0)
        
    final_score.append(1 if sum(def_score) >= 2 else 0) #sum values in def_score and append to final_score
    
df_combined_agg['Defensive Effort'] = final_score
```


```python
df_combined_agg.columns, df_def_agg.columns, df_off_agg.columns
```




    (Index(['game_id', 'Team_Abbrev', 'Opponent_Abbrev', 'outcome', 'sumfg',
            'sumfg3', 'sumft', 'sumorb', 'sumast', 'sumpts', 'sumdrb', 'sumstl',
            'sumblk', 'opponent_id', 'Offensive Effort', 'Defensive Effort'],
           dtype='object'),
     Index(['game_id', 'Team_Abbrev', 'Opponent_Abbrev', 'outcome', 'sumdrb',
            'sumstl', 'sumblk'],
           dtype='object'),
     Index(['game_id', 'Team_Abbrev', 'Opponent_Abbrev', 'outcome', 'sumfg',
            'sumfg3', 'sumft', 'sumorb', 'sumast', 'sumpts'],
           dtype='object'))




```python
#get subset of data that only contains fields with numerical values

cols_to_exclude = ['game_id','Team_Abbrev','Opponent_Abbrev']

df_def_num = df_def_agg.drop(columns=cols_to_exclude)
df_off_num = df_off_agg.drop(columns=cols_to_exclude)
df_combined_num = df_combined_agg.drop(columns=cols_to_exclude)
df_combined_num = df_combined_num.drop(columns='opponent_id')
```


```python
df_combined_num.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>outcome</th>
      <th>sumfg</th>
      <th>sumfg3</th>
      <th>sumft</th>
      <th>sumorb</th>
      <th>sumast</th>
      <th>sumpts</th>
      <th>sumdrb</th>
      <th>sumstl</th>
      <th>sumblk</th>
      <th>Offensive Effort</th>
      <th>Defensive Effort</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>69</td>
      <td>20</td>
      <td>31</td>
      <td>17</td>
      <td>44</td>
      <td>189</td>
      <td>61</td>
      <td>11</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>67</td>
      <td>23</td>
      <td>29</td>
      <td>17</td>
      <td>39</td>
      <td>186</td>
      <td>60</td>
      <td>8</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>43</td>
      <td>19</td>
      <td>17</td>
      <td>16</td>
      <td>30</td>
      <td>122</td>
      <td>37</td>
      <td>4</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>76</td>
      <td>25</td>
      <td>48</td>
      <td>29</td>
      <td>40</td>
      <td>225</td>
      <td>75</td>
      <td>11</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>64</td>
      <td>20</td>
      <td>16</td>
      <td>10</td>
      <td>25</td>
      <td>164</td>
      <td>49</td>
      <td>13</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Data Exploration


```python
#get subset of numerated columns in preparation for correlation matrix

# Normalize quantitative fields using Min-Max scaling
scaler = MinMaxScaler()
df_def_scaled = pd.DataFrame(scaler.fit_transform(df_def_num), columns=df_def_num.columns)
df_off_scaled = pd.DataFrame(scaler.fit_transform(df_off_num), columns=df_off_num.columns)
df_combined_scaled = pd.DataFrame(scaler.fit_transform(df_combined_num), columns=df_combined_num.columns)



scaled_df_lst = [df_def_scaled,df_off_scaled,df_combined_scaled]

for df in scaled_df_lst:
    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    plt.figure(figsize=(5,5))

    # Create a heatmap using seaborn
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.5)



    # Show the plot
    plt.show()
```


    
![png](output_30_0.png)
    



    
![png](output_30_1.png)
    



    
![png](output_30_2.png)
    



```python
#distribution of wins and losses
for df in scaled_df_lst:

    sns.countplot(x='outcome', data=df)

    plt.show()
```

    C:\Users\theri\anaconda3\lib\site-packages\seaborn\_core.py:1225: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\theri\anaconda3\lib\site-packages\seaborn\_core.py:1225: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\theri\anaconda3\lib\site-packages\seaborn\_core.py:1225: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    


    
![png](output_31_1.png)
    


    C:\Users\theri\anaconda3\lib\site-packages\seaborn\_core.py:1225: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\theri\anaconda3\lib\site-packages\seaborn\_core.py:1225: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\theri\anaconda3\lib\site-packages\seaborn\_core.py:1225: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    


    
![png](output_31_3.png)
    


    C:\Users\theri\anaconda3\lib\site-packages\seaborn\_core.py:1225: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\theri\anaconda3\lib\site-packages\seaborn\_core.py:1225: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    C:\Users\theri\anaconda3\lib\site-packages\seaborn\_core.py:1225: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    


    
![png](output_31_5.png)
    



```python
#visualize the effect of defense and offense on game outcome

# Group by 'offense' and 'defense', then count the occurrences of each outcome
grouped_df = df_combined_scaled.groupby(['Offensive Effort', 'Defensive Effort', 'outcome']).size().reset_index(name='count')

# Pivot the dataframe to make 'outcome' values as columns
pivot_df = grouped_df.pivot_table(index=['Offensive Effort', 'Defensive Effort'], columns='outcome', fill_value=0)

# Plot the stacked bar chart
pivot_df.plot(kind='bar', stacked=False)
plt.title('Outcome Count by Offensive and Defensive Effort')
plt.xlabel('Offensive Effort, Defensive Effort')
plt.ylabel('Count')
plt.legend(title='Outcome')
plt.show()
```


    
![png](output_32_0.png)
    


## 3. Define & Train Model


```python
#import ML libraries
from sklearn.svm import SVC


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm as svm_linear
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split as tts

```


```python
def misclassification_cost(arg1, arg2):
    
    # function for calculating the missclassification cost of a classifier given test labels and predicted labels returned from the
    # trained classifier.
    
    '''
    inputs:
    arg1 = array of test labels
    arg2 = array of predicted labels
    
    returns:
    misclassification cost
    '''

    #print confusion matrix
    CM = confusion_matrix(arg1,arg2)
    print(CM)

    #define cost matrix shape
    cM = np.zeros(CM.shape)

    #assign weights to cost matrix
    if cM.shape == (3,3):
        cM[0] = [0,1,2]
        cM[1] = [1,0,1]
        cM[2] = [2,1,0]

    # for binary classification
    elif cM.shape ==(2,2):
        cM[0] = [0,1]
        cM[1] = [1,0]

    elif cM.shape == (5,5):
        cM[0] = [0,1,2,3,4]
        cM[1] = [1,0,1,2,3]
        cM[2] = [2,1,0,1,2]
        cM[3] = [3,2,1,0,1]
        cM[4] = [4,3,2,1,0]

    #calculate classification cost
    cM_matrix = np.matrix(CM * cM)
    clcost = cM_matrix.sum()/arg2.shape[0]
    
    return(clcost)
```


```python
def classifier(arg1,arg2,arg3):
    
    '''
    arg1: X Features
    arg2: y Label
    arg3: random seed
    '''


    modelDict = {'DecisionTreeClassifier':'max_depth = %d,random_state = %d' % (4,0),
                 'RandomForestClassifier':'',
                 'svm.SVC':'kernel="rbf"', 
                 'KNeighborsClassifier':'n_neighbors = 4',
                 'svm_linear.SVC':'kernel="linear"'
                }

    model_lst = list(modelDict.keys())  


    #create lists for storing model scores for cross-validation
    models = []
    accuracy = []
    mpca = []
    f1_wt = []

    #create lists for storing model scores
    accuracy2 = []
    mpca2 = []
    f1_wt2 = []
    cost2 = []
    unique2 = []



    for key in modelDict.keys():
        print(key)
        models.append(key)

        #define classifier with parameters including penalizing parameters
        clf = eval('%s(%s)' % (key,modelDict[key]))


        #evaluate pipeline

        # define cross-validation method for model evaluation
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)       


        #calculate cross-validated mean per class accuracy (recall macroaverage from classification report)
        results = cross_validate(clf,arg1, arg2, scoring=['recall_macro','accuracy','f1_macro','f1_weighted'], cv=cv, n_jobs=-1,error_score='raise')

        #append cross-validation results to respective lists
        accuracy.append(results['test_accuracy'].mean())
        mpca.append(results['test_recall_macro'].mean())
        f1_wt.append(results['test_f1_weighted'].mean())





        #train each model with training data and predict on test data

        #split data into train and test data (75/25 split)
        X_train, X_test, y_train, y_test = tts(arg1, arg2, random_state=arg3)


        print(np.unique(y_train))
        #train models
        model = clf.fit(X_train, y_train)  


        #predict on test data
        y_hat = model.predict(X_test)      
        print(classification_report(y_test, y_hat))

        #print confusion matrix
        CM = confusion_matrix(y_test,y_hat)
        print(CM)


        #capture classification accuracy metrics
        report_dict = classification_report(y_test,y_hat,output_dict=True)

        #mean per class accuracy
        mpca2.append(report_dict['macro avg']['recall']) #mean per class accuracy

        #return f1 score
        f1_wt2.append(report_dict['weighted avg']['f1-score'])

        #accuracy
        accuracy2.append(report_dict['accuracy'])   

        #misclassification cost
        cost2.append(misclassification_cost(y_test,y_hat))

        #unique label predictions
        unique2.append(np.unique(y_hat))



    #create and populate dataframe with cross-validation results
    df_scores = pd.DataFrame()
    df_scores['Model'] = models
    df_scores['CV Accuracy'] = accuracy
    df_scores['CV MPCA'] = mpca
    df_scores['CV F1_weighted'] = f1_wt
    
    
    df_scores2 = pd.DataFrame()
    df_scores2['Model'] = models
    df_scores2['Accuracy'] = accuracy2
    df_scores2['MPCA'] = mpca2
    df_scores2['F1_weighted'] = f1_wt2
    df_scores2['Misclassification_Cost'] = cost2
    df_scores2['unique predictions'] = unique2




    return(df_scores,df_scores2,model)
```


```python
#Define features and labels for all datasets
X_def = df_def_scaled.drop(columns= ['outcome'],axis=1) #predictors
y_def = df_def_scaled['outcome']

X_off = df_off_scaled.drop(columns= ['outcome'],axis=1) #predictors
y_off = df_off_scaled['outcome']

X_com = df_combined_scaled.drop(columns= ['outcome'],axis=1) #predictors
y_com = df_combined_scaled['outcome']

X_com_defeffort = df_combined_scaled.drop(columns= ['outcome','sumfg','sumfg3',
                                              'sumft','sumorb','sumast','sumstl','sumblk','sumpts',
                                              'Offensive Effort'],axis=1) #predictors
X_com_offeffort = df_combined_scaled.drop(columns= ['outcome','sumfg','sumfg3',
                                              'sumft','sumorb','sumast','sumstl','sumblk','sumpts',
                                              'Defensive Effort'],axis=1) #predictors
```


```python
X_com.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sumfg</th>
      <th>sumfg3</th>
      <th>sumft</th>
      <th>sumorb</th>
      <th>sumast</th>
      <th>sumpts</th>
      <th>sumdrb</th>
      <th>sumstl</th>
      <th>sumblk</th>
      <th>Offensive Effort</th>
      <th>Defensive Effort</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.484211</td>
      <td>0.346154</td>
      <td>0.337349</td>
      <td>0.404762</td>
      <td>0.400000</td>
      <td>0.481328</td>
      <td>0.548780</td>
      <td>0.333333</td>
      <td>0.212121</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.463158</td>
      <td>0.403846</td>
      <td>0.313253</td>
      <td>0.404762</td>
      <td>0.344444</td>
      <td>0.468880</td>
      <td>0.536585</td>
      <td>0.242424</td>
      <td>0.363636</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.210526</td>
      <td>0.326923</td>
      <td>0.168675</td>
      <td>0.380952</td>
      <td>0.244444</td>
      <td>0.203320</td>
      <td>0.256098</td>
      <td>0.121212</td>
      <td>0.272727</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.557895</td>
      <td>0.442308</td>
      <td>0.542169</td>
      <td>0.690476</td>
      <td>0.355556</td>
      <td>0.630705</td>
      <td>0.719512</td>
      <td>0.333333</td>
      <td>0.181818</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.431579</td>
      <td>0.346154</td>
      <td>0.156627</td>
      <td>0.238095</td>
      <td>0.188889</td>
      <td>0.377593</td>
      <td>0.402439</td>
      <td>0.393939</td>
      <td>0.303030</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
comval,commdl,mdl = classifier(X_com_offeffort,y_com,42)
```

    DecisionTreeClassifier
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.74      0.81      0.77       907
             1.0       0.78      0.70      0.74       865
    
        accuracy                           0.76      1772
       macro avg       0.76      0.75      0.75      1772
    weighted avg       0.76      0.76      0.76      1772
    
    [[733 174]
     [258 607]]
    [[733 174]
     [258 607]]
    RandomForestClassifier
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.75      0.78      0.76       907
             1.0       0.76      0.72      0.74       865
    
        accuracy                           0.75      1772
       macro avg       0.75      0.75      0.75      1772
    weighted avg       0.75      0.75      0.75      1772
    
    [[708 199]
     [242 623]]
    [[708 199]
     [242 623]]
    svm.SVC
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.74      0.81      0.77       907
             1.0       0.78      0.70      0.74       865
    
        accuracy                           0.76      1772
       macro avg       0.76      0.76      0.76      1772
    weighted avg       0.76      0.76      0.76      1772
    
    [[733 174]
     [256 609]]
    [[733 174]
     [256 609]]
    KNeighborsClassifier
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.68      0.86      0.76       907
             1.0       0.80      0.58      0.68       865
    
        accuracy                           0.73      1772
       macro avg       0.74      0.72      0.72      1772
    weighted avg       0.74      0.73      0.72      1772
    
    [[781 126]
     [360 505]]
    [[781 126]
     [360 505]]
    svm_linear.SVC
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.74      0.81      0.77       907
             1.0       0.78      0.70      0.74       865
    
        accuracy                           0.76      1772
       macro avg       0.76      0.76      0.76      1772
    weighted avg       0.76      0.76      0.76      1772
    
    [[733 174]
     [256 609]]
    [[733 174]
     [256 609]]
    


```python
comval
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>CV Accuracy</th>
      <th>CV MPCA</th>
      <th>CV F1_weighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DecisionTreeClassifier</td>
      <td>0.797309</td>
      <td>0.797305</td>
      <td>0.796798</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RandomForestClassifier</td>
      <td>0.809396</td>
      <td>0.809397</td>
      <td>0.809344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>svm.SVC</td>
      <td>0.799660</td>
      <td>0.799662</td>
      <td>0.798967</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNeighborsClassifier</td>
      <td>0.786915</td>
      <td>0.786918</td>
      <td>0.785422</td>
    </tr>
    <tr>
      <th>4</th>
      <td>svm_linear.SVC</td>
      <td>0.797261</td>
      <td>0.797265</td>
      <td>0.795974</td>
    </tr>
  </tbody>
</table>
</div>




```python
commdl
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>MPCA</th>
      <th>F1_weighted</th>
      <th>Misclassification_Cost</th>
      <th>unique predictions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DecisionTreeClassifier</td>
      <td>0.791196</td>
      <td>0.789848</td>
      <td>0.790372</td>
      <td>0.208804</td>
      <td>[0.0, 1.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RandomForestClassifier</td>
      <td>0.804176</td>
      <td>0.803410</td>
      <td>0.803897</td>
      <td>0.195824</td>
      <td>[0.0, 1.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>svm.SVC</td>
      <td>0.797404</td>
      <td>0.795671</td>
      <td>0.796121</td>
      <td>0.202596</td>
      <td>[0.0, 1.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNeighborsClassifier</td>
      <td>0.775395</td>
      <td>0.773342</td>
      <td>0.773458</td>
      <td>0.224605</td>
      <td>[0.0, 1.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>svm_linear.SVC</td>
      <td>0.791196</td>
      <td>0.788938</td>
      <td>0.789019</td>
      <td>0.208804</td>
      <td>[0.0, 1.0]</td>
    </tr>
  </tbody>
</table>
</div>




```python
defval,defmdl,mdl = classifier(X_def,y_def,42)
```

    DecisionTreeClassifier
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.71      0.51      0.59       907
             1.0       0.60      0.78      0.68       865
    
        accuracy                           0.64      1772
       macro avg       0.65      0.64      0.64      1772
    weighted avg       0.66      0.64      0.63      1772
    
    [[463 444]
     [192 673]]
    [[463 444]
     [192 673]]
    RandomForestClassifier
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.62      0.62      0.62       907
             1.0       0.60      0.61      0.60       865
    
        accuracy                           0.61      1772
       macro avg       0.61      0.61      0.61      1772
    weighted avg       0.61      0.61      0.61      1772
    
    [[559 348]
     [339 526]]
    [[559 348]
     [339 526]]
    svm.SVC
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.68      0.55      0.61       907
             1.0       0.61      0.73      0.66       865
    
        accuracy                           0.64      1772
       macro avg       0.64      0.64      0.64      1772
    weighted avg       0.64      0.64      0.64      1772
    
    [[501 406]
     [236 629]]
    [[501 406]
     [236 629]]
    KNeighborsClassifier
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.58      0.74      0.65       907
             1.0       0.62      0.44      0.51       865
    
        accuracy                           0.59      1772
       macro avg       0.60      0.59      0.58      1772
    weighted avg       0.60      0.59      0.58      1772
    
    [[674 233]
     [486 379]]
    [[674 233]
     [486 379]]
    svm_linear.SVC
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.57      0.75      0.65       907
             1.0       0.61      0.40      0.48       865
    
        accuracy                           0.58      1772
       macro avg       0.59      0.58      0.57      1772
    weighted avg       0.59      0.58      0.57      1772
    
    [[681 226]
     [516 349]]
    [[681 226]
     [516 349]]
    


```python
offval,offmdl,mdl = classifier(X_off,y_off,42)
```

    DecisionTreeClassifier
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.70      0.66      0.68       907
             1.0       0.66      0.70      0.68       865
    
        accuracy                           0.68      1772
       macro avg       0.68      0.68      0.68      1772
    weighted avg       0.68      0.68      0.68      1772
    
    [[597 310]
     [256 609]]
    [[597 310]
     [256 609]]
    RandomForestClassifier
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.68      0.66      0.67       907
             1.0       0.65      0.67      0.66       865
    
        accuracy                           0.66      1772
       macro avg       0.66      0.66      0.66      1772
    weighted avg       0.66      0.66      0.66      1772
    
    [[596 311]
     [285 580]]
    [[596 311]
     [285 580]]
    svm.SVC
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.73      0.62      0.67       907
             1.0       0.65      0.75      0.70       865
    
        accuracy                           0.68      1772
       macro avg       0.69      0.69      0.68      1772
    weighted avg       0.69      0.68      0.68      1772
    
    [[560 347]
     [212 653]]
    [[560 347]
     [212 653]]
    KNeighborsClassifier
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.60      0.75      0.67       907
             1.0       0.65      0.48      0.55       865
    
        accuracy                           0.62      1772
       macro avg       0.62      0.61      0.61      1772
    weighted avg       0.62      0.62      0.61      1772
    
    [[681 226]
     [454 411]]
    [[681 226]
     [454 411]]
    svm_linear.SVC
    [0. 1.]
                  precision    recall  f1-score   support
    
             0.0       0.58      0.76      0.65       907
             1.0       0.62      0.41      0.49       865
    
        accuracy                           0.59      1772
       macro avg       0.60      0.59      0.57      1772
    weighted avg       0.60      0.59      0.58      1772
    
    [[689 218]
     [509 356]]
    [[689 218]
     [509 356]]
    


```python
defval
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>CV Accuracy</th>
      <th>CV MPCA</th>
      <th>CV F1_weighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DecisionTreeClassifier</td>
      <td>0.645271</td>
      <td>0.645269</td>
      <td>0.642594</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RandomForestClassifier</td>
      <td>0.616349</td>
      <td>0.616352</td>
      <td>0.616174</td>
    </tr>
    <tr>
      <th>2</th>
      <td>svm.SVC</td>
      <td>0.654675</td>
      <td>0.654673</td>
      <td>0.650606</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNeighborsClassifier</td>
      <td>0.610048</td>
      <td>0.610047</td>
      <td>0.601888</td>
    </tr>
    <tr>
      <th>4</th>
      <td>svm_linear.SVC</td>
      <td>0.592081</td>
      <td>0.592080</td>
      <td>0.581819</td>
    </tr>
  </tbody>
</table>
</div>




```python
defmdl
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>MPCA</th>
      <th>F1_weighted</th>
      <th>Misclassification_Cost</th>
      <th>unique predictions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DecisionTreeClassifier</td>
      <td>0.641084</td>
      <td>0.644254</td>
      <td>0.634948</td>
      <td>0.358916</td>
      <td>[0.0, 1.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RandomForestClassifier</td>
      <td>0.612302</td>
      <td>0.612205</td>
      <td>0.612339</td>
      <td>0.387698</td>
      <td>[0.0, 1.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>svm.SVC</td>
      <td>0.637698</td>
      <td>0.639769</td>
      <td>0.635174</td>
      <td>0.362302</td>
      <td>[0.0, 1.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNeighborsClassifier</td>
      <td>0.594244</td>
      <td>0.590630</td>
      <td>0.584324</td>
      <td>0.405756</td>
      <td>[0.0, 1.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>svm_linear.SVC</td>
      <td>0.581264</td>
      <td>0.577148</td>
      <td>0.567957</td>
      <td>0.418736</td>
      <td>[0.0, 1.0]</td>
    </tr>
  </tbody>
</table>
</div>




```python
offval
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>CV Accuracy</th>
      <th>CV MPCA</th>
      <th>CV F1_weighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DecisionTreeClassifier</td>
      <td>0.680822</td>
      <td>0.680807</td>
      <td>0.679296</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RandomForestClassifier</td>
      <td>0.656932</td>
      <td>0.656924</td>
      <td>0.656740</td>
    </tr>
    <tr>
      <th>2</th>
      <td>svm.SVC</td>
      <td>0.679128</td>
      <td>0.679125</td>
      <td>0.678037</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNeighborsClassifier</td>
      <td>0.622555</td>
      <td>0.622550</td>
      <td>0.616651</td>
    </tr>
    <tr>
      <th>4</th>
      <td>svm_linear.SVC</td>
      <td>0.594715</td>
      <td>0.594713</td>
      <td>0.584161</td>
    </tr>
  </tbody>
</table>
</div>




```python
offmdl
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>MPCA</th>
      <th>F1_weighted</th>
      <th>Misclassification_Cost</th>
      <th>unique predictions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DecisionTreeClassifier</td>
      <td>0.680587</td>
      <td>0.681130</td>
      <td>0.680521</td>
      <td>0.319413</td>
      <td>[0.0, 1.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RandomForestClassifier</td>
      <td>0.663657</td>
      <td>0.663816</td>
      <td>0.663701</td>
      <td>0.336343</td>
      <td>[0.0, 1.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>svm.SVC</td>
      <td>0.684537</td>
      <td>0.686167</td>
      <td>0.683272</td>
      <td>0.315463</td>
      <td>[0.0, 1.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNeighborsClassifier</td>
      <td>0.616253</td>
      <td>0.612986</td>
      <td>0.608551</td>
      <td>0.383747</td>
      <td>[0.0, 1.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>svm_linear.SVC</td>
      <td>0.589729</td>
      <td>0.585604</td>
      <td>0.576604</td>
      <td>0.410271</td>
      <td>[0.0, 1.0]</td>
    </tr>
  </tbody>
</table>
</div>



# Save Model


```python
import joblib

joblib.dump(mdl,'nba_outcome.joblib')
```




    ['nba_outcome.joblib']



# Supplemental Code


```python
'''
for df in scaled_df_lst:
    

    #define predictor and response variables
    X = df.drop(columns= ['outcome'],axis=1) #predictors
    y = df['outcome']              #response

    # convert class label column to int type
    y = y.astype(np.int_)

    from sklearn.model_selection import train_test_split as tts

    X_train, X_test, y_train, y_test = tts(X,y,random_state=3) 

    from collections import Counter

    train_count = Counter(y_train)
    test_count = Counter(y_test)

    print(train_count)
    print(test_count)


    #define model
    svc = SVC(kernel='rbf')

    model = svc.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy on test set: {accuracy}")

    from sklearn.metrics import classification_report

    #Precision, recall, f1-score and support
    print(classification_report(y_test, y_pred))
'''
```

    Counter({0: 2692, 1: 2624})
    Counter({1: 920, 0: 852})
    Accuracy on test set: 0.6360045146726863
                  precision    recall  f1-score   support
    
               0       0.65      0.54      0.59       852
               1       0.63      0.73      0.67       920
    
        accuracy                           0.64      1772
       macro avg       0.64      0.63      0.63      1772
    weighted avg       0.64      0.64      0.63      1772
    
    Counter({0: 2692, 1: 2624})
    Counter({1: 920, 0: 852})
    Accuracy on test set: 0.6783295711060948
                  precision    recall  f1-score   support
    
               0       0.68      0.62      0.65       852
               1       0.68      0.73      0.70       920
    
        accuracy                           0.68      1772
       macro avg       0.68      0.68      0.68      1772
    weighted avg       0.68      0.68      0.68      1772
    
    Counter({0: 2692, 1: 2624})
    Counter({1: 920, 0: 852})
    Accuracy on test set: 0.79627539503386
                  precision    recall  f1-score   support
    
               0       0.75      0.86      0.80       852
               1       0.85      0.74      0.79       920
    
        accuracy                           0.80      1772
       macro avg       0.80      0.80      0.80      1772
    weighted avg       0.80      0.80      0.80      1772
    
    


```python
'''
#Find what error value is in the 'DKP_per_minute' field
for i in range(0,df_full.shape[0]):
    try:
        float(df_full['DKP_per_minute'][i])      
    except ValueError as e:
        print(df_full['DKP_per_minute'][i])
        print(df_full.iloc[i])
        print(e)
        
'''
```

    #NAME?
    game_id                                     201912260SAC
    game_date                            2019-12-26 00:00:00
    OT                                                     2
    H_A                                                    H
    Team_Abbrev                                          SAC
    Team_Score                                           104
    Team_pace                                           92.7
    Team_efg_pct                                       0.388
    Team_tov_pct                                         9.2
    Team_orb_pct                                        23.9
    Team_ft_rate                                       0.196
    Team_off_rtg                                        92.8
    Opponent_Abbrev                                      MIN
    Opponent_Score                                       105
    Opponent_pace                                       92.7
    Opponent_efg_pct                                   0.394
    Opponent_tov_pct                                    10.0
    Opponent_orb_pct                                    21.5
    Opponent_ft_rate                                   0.221
    Opponent_off_rtg                                    93.7
    player                                      Justin James
    player_id                                      jamesju01
    starter                                                0
    mp                                                  0:00
    fg                                                     0
    fga                                                    0
    fg_pct                                               0.0
    fg3                                                    0
    fg3a                                                   0
    fg3_pct                                              0.0
    ft                                                     0
    fta                                                    0
    ft_pct                                               0.0
    orb                                                    0
    drb                                                    0
    trb                                                    0
    ast                                                    0
    stl                                                    0
    blk                                                    0
    tov                                                    1
    pf                                                     0
    pts                                                    0
    plus_minus                                             0
    did_not_play                                           0
    is_inactive                                            0
    ts_pct                                               0.0
    efg_pct                                              0.0
    fg3a_per_fga_pct                                     0.0
    fta_per_fga_pct                                      0.0
    orb_pct                                              0.0
    drb_pct                                              0.0
    trb_pct                                              0.0
    ast_pct                                              0.0
    stl_pct                                              0.0
    blk_pct                                              0.0
    tov_pct                                            100.0
    usg_pct                                              0.0
    off_rtg                                                0
    def_rtg                                                0
    bpm                                                 -8.9
    season                                              2020
    minutes                                              0.0
    double_double                                          0
    triple_double                                          0
    DKP                                                 -0.5
    FDP                                                 -1.0
    SDP                                                 -1.0
    DKP_per_minute                                    #NAME?
    FDP_per_minute                                    #NAME?
    SDP_per_minute                                    #NAME?
    pf_per_minute                                        NaN
    ts                                                   0.0
    last_60_minutes_per_game_starting               10.38033
    last_60_minutes_per_game_bench                  6.029487
    PG%                                                  0.0
    SG%                                                 19.0
    SF%                                                 70.0
    PF%                                                 10.0
    C%                                                   1.0
    active_position_minutes                        57.047569
    duplicate                                          False
    outcome                                                0
    no_inactives                                           4
    min_played                                             0
    Name: 105768, dtype: object
    could not convert string to float: '#NAME?'
    
