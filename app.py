 naitik429-patch-1
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "fd45fc87-50e1-4097-b874-7495f7df48c6",
   "metadata": {},
   "source": [
    "# Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0a5a2ea2-7182-4aa4-90a8-5277b5ae3f07",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7c132480-0714-40bc-bc17-542dd04b6e56",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>887</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Montvila, Rev. Juozas</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>211536</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>888</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Graham, Miss. Margaret Edith</td>\n",
       "      <td>female</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>112053</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>B42</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>889</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Johnston, Miss. Catherine Helen \"Carrie\"</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>W./C. 6607</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>890</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Behr, Mr. Karl Howell</td>\n",
       "      <td>male</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>111369</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>C148</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>891</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Dooley, Mr. Patrick</td>\n",
       "      <td>male</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>370376</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass  \\\n",
       "0              1         0       3   \n",
       "1              2         1       1   \n",
       "2              3         1       3   \n",
       "3              4         1       1   \n",
       "4              5         0       3   \n",
       "..           ...       ...     ...   \n",
       "886          887         0       2   \n",
       "887          888         1       1   \n",
       "888          889         0       3   \n",
       "889          890         1       1   \n",
       "890          891         0       3   \n",
       "\n",
       "                                                  Name     Sex   Age  SibSp  \\\n",
       "0                              Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                               Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                             Allen, Mr. William Henry    male  35.0      0   \n",
       "..                                                 ...     ...   ...    ...   \n",
       "886                              Montvila, Rev. Juozas    male  27.0      0   \n",
       "887                       Graham, Miss. Margaret Edith  female  19.0      0   \n",
       "888           Johnston, Miss. Catherine Helen \"Carrie\"  female   NaN      1   \n",
       "889                              Behr, Mr. Karl Howell    male  26.0      0   \n",
       "890                                Dooley, Mr. Patrick    male  32.0      0   \n",
       "\n",
       "     Parch            Ticket     Fare Cabin Embarked  \n",
       "0        0         A/5 21171   7.2500   NaN        S  \n",
       "1        0          PC 17599  71.2833   C85        C  \n",
       "2        0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3        0            113803  53.1000  C123        S  \n",
       "4        0            373450   8.0500   NaN        S  \n",
       "..     ...               ...      ...   ...      ...  \n",
       "886      0            211536  13.0000   NaN        S  \n",
       "887      0            112053  30.0000   B42        S  \n",
       "888      2        W./C. 6607  23.4500   NaN        S  \n",
       "889      0            111369  30.0000  C148        C  \n",
       "890      0            370376   7.7500   NaN        Q  \n",
       "\n",
       "[891 rows x 12 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('Titanic_train.csv')\n",
    "df\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4c39e170-351b-4603-a36c-78106fbcff7c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>female</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>1305</td>\n",
       "      <td>3</td>\n",
       "      <td>Spector, Mr. Woolf</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>A.5. 3236</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>1306</td>\n",
       "      <td>1</td>\n",
       "      <td>Oliva y Ocana, Dona. Fermina</td>\n",
       "      <td>female</td>\n",
       "      <td>39.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17758</td>\n",
       "      <td>108.9000</td>\n",
       "      <td>C105</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>415</th>\n",
       "      <td>1307</td>\n",
       "      <td>3</td>\n",
       "      <td>Saether, Mr. Simon Sivertsen</td>\n",
       "      <td>male</td>\n",
       "      <td>38.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>SOTON/O.Q. 3101262</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>416</th>\n",
       "      <td>1308</td>\n",
       "      <td>3</td>\n",
       "      <td>Ware, Mr. Frederick</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>359309</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>1309</td>\n",
       "      <td>3</td>\n",
       "      <td>Peter, Master. Michael J</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2668</td>\n",
       "      <td>22.3583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>418 rows × 11 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Pclass                                          Name  \\\n",
       "0            892       3                              Kelly, Mr. James   \n",
       "1            893       3              Wilkes, Mrs. James (Ellen Needs)   \n",
       "2            894       2                     Myles, Mr. Thomas Francis   \n",
       "3            895       3                              Wirz, Mr. Albert   \n",
       "4            896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)   \n",
       "..           ...     ...                                           ...   \n",
       "413         1305       3                            Spector, Mr. Woolf   \n",
       "414         1306       1                  Oliva y Ocana, Dona. Fermina   \n",
       "415         1307       3                  Saether, Mr. Simon Sivertsen   \n",
       "416         1308       3                           Ware, Mr. Frederick   \n",
       "417         1309       3                      Peter, Master. Michael J   \n",
       "\n",
       "        Sex   Age  SibSp  Parch              Ticket      Fare Cabin Embarked  \n",
       "0      male  34.5      0      0              330911    7.8292   NaN        Q  \n",
       "1    female  47.0      1      0              363272    7.0000   NaN        S  \n",
       "2      male  62.0      0      0              240276    9.6875   NaN        Q  \n",
       "3      male  27.0      0      0              315154    8.6625   NaN        S  \n",
       "4    female  22.0      1      1             3101298   12.2875   NaN        S  \n",
       "..      ...   ...    ...    ...                 ...       ...   ...      ...  \n",
       "413    male   NaN      0      0           A.5. 3236    8.0500   NaN        S  \n",
       "414  female  39.0      0      0            PC 17758  108.9000  C105        C  \n",
       "415    male  38.5      0      0  SOTON/O.Q. 3101262    7.2500   NaN        S  \n",
       "416    male   NaN      0      0              359309    8.0500   NaN        S  \n",
       "417    male   NaN      1      1                2668   22.3583   NaN        C  \n",
       "\n",
       "[418 rows x 11 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2 = pd.read_csv('Titanic_test.csv')\n",
    "df2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "057177d1-0387-4a5f-9a46-e027674ff3c6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    str    \n",
      " 4   Sex          891 non-null    str    \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    str    \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    str    \n",
      " 11  Embarked     889 non-null    str    \n",
      "dtypes: float64(2), int64(5), str(5)\n",
      "memory usage: 118.9 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "342f423b-5df5-4857-b1af-1dcab23fce2d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "124b309a-171e-447f-9ac4-b0e979ade739",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   38.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a2e78a7c-e8cf-4b04-954b-62acdb94dc63",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "af4d5645-c7c4-4fb8-ac83-6469cba7f695",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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hlGjRqF///7Ys2dPvgVGRERE9DmCUHi3r1GuksDDhw/Dx8dHfj8pKQmNGjXC5s2bcefOHXTv3h07duzI9yCJiIiIKH/lKglcuHChwiVT9u7di7dv3yI4OBiBgYGYP38+Fi9enO9BEhEREX2KlcC8yVUSePXqVdSoUUN+//Tp0/D29oaDgwMAoG/fvrh//37+RkhERERE+S5XSaBEIkFSUpL8/qVLl1CnTh35fT09PYXVr4mIiIgKikyQFNrta5SrJLBatWqYN28eBEHA2bNncfv2bbRo0UL++N27d1GhQoV8D5KIiIiI8leurkI9depUeHt7Y+nSpUhOTkaHDh1QpUoV+eObN29G//798z1IIiIiok99rWP1CkuuksAGDRogKCgIfn5+sLKyQvfu3RUer1y5Mnr37p2vARIRERFR/stVEggAbm5ucHNzy/ax7777Ls8BERERESmDlcC8KZwrFBMRERGRWsl1JZCIiIhIHfDawXnDSiARERGRBmIlkIiIiIok4Stdv6+wsBJIREREpIGYBBIRERFpIHYHExERUZHEJWLyhpVAIiIiIg3ESiAREREVSVwiJm9YCSQiIiLSQKwEEhERUZHEMYF5w0ogERERkQZiJZCIiIiKJFYC84aVQCIiIiINxEogERERFUmcHZw3rAQSERERaSBWAomIiKhI4pjAvGElkIiIiEgDqU0l0MRAJnYIasvKKEXsENRWXIqe2CGoLYeSBmKHoJaa/DdL7BDU1pnGE8QOQW253zsqdgiUDRlThzxhJZCIiIhIA6lNJZCIiIgoNzgmMG9YCSQiIiLSQKwEEhERUZHESmDesBJIREREpIGYBBIRERFpIHYHExERUZHEy8blDSuBRERERBqIlUAiIiIqkoRCnRkiKcTnKhysBBIRERFpIFYCiYiIqEjiEjF5w0ogERERkQZiJZCIiIiKJJlM7AiKNlYCiYiIiDQQK4FERERUJHFMYN6wEkhERESkgVgJJCIioiKJVwzJG1YCiYiIiDQQK4FERERUJHFMYN6olATGx8dj9uzZuHDhAqKiorI8HhQUlNe4iIiIiKgAqZQEfv/99/D390fv3r1haWmZ3zERERERfZFQqIMCv75rB6uUBB45cgQBAQFwd3fP73iIiIiIqBColASampqiWLFi+R0LERERkdI4OzhvVJod3KlTJ6xYsSK/YyEi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",
      "text/plain": [
       "<Figure size 800x700 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (8,7))\n",
    "\n",
    "sns.heatmap(df.corr(numeric_only = True), annot = True, cmap = 'coolwarm')\n",
    "plt.title('Correlation between numeric values')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "49d23bce-e80e-4d80-9b7a-84fa2a0c56fd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,6))\n",
    "sns.heatmap(df.corr(numeric_only = True), cmap = 'coolwarm', annot = True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9079ac9f-6826-464c-945a-a6e2932f537f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 800x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,6))\n",
    "plt.subplot(2,2,1)\n",
    "sns.boxplot(data = df, x = 'Survived', y = 'Age')\n",
    "plt.show()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "eadac0a5-5683-4a06-ae31-3f9d6069d4cd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAh8AAAGyCAYAAABEN6Z8AAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlcelbwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAO9dJREFUeJzt3XlclOX+//H3sDgKiOJSoKKmcdLEfSvTrGNopmaLZlFpanbaPJlRaYuhHs1TUXaqY4ucSsu9MlHL0NTM1BYttzTTUtJcIFmUxBGu3x/9vL9OgDIw3DDT6/l48LD7muu+5vOZGYZ399wz4zDGGAEAANgkoKILAAAAfy2EDwAAYCvCBwAAsBXhAwAA2IrwAQAAbEX4AAAAtiJ8AAAAWxE+AACArQgfACqtgwcPKiEhQd9++6019vPPPyshIUHff/99hdYBoPQIH4CNTp48qUceeUQPP/ywjh07VqG1ZGVlKSEhwfoZM2aMJk2apHfeeUe7du066z6LFi3y6Lp++eUXJSQkaMuWLR7tl56erqSkJO3YscNtraSkJO3evdujtcpSY1F1ACg9wgdgo4ULF+rZZ5/Vc889p9mzZ1doLTk5OUpKStKXX36pyMhI1a5dW7///rsWLFigli1b6rLLLiv0f/qn9/n00089uq6DBw8qKSlJO3fu9Gi/qKgoPfvss2rbtq1H+5XG2Wq0sw7gryCoogsA/kqmT5+u9u3bq0aNGpo+fbpGjBhR0SWpXbt2SkhIcBvbv3+/rr/+enXr1k1r1qxRmzZtKqS22rVrF6rtr1wH4C8IH4BNfv75Zy1fvlzTp09XjRo1NGDAAG3evFmtWrUqcv6aNWu0YsUKOZ1O3XDDDWrUqJGeeOIJ9enTR1deeaXb3F9//VWLFi3Svn37VKNGDV1zzTWKjY0tda3169fXe++9pwsvvFAPPvigVq5cedb56enpWrhwofbt26fatWura9euat++vSRpy5YtevHFFyVJM2bM0Pr16yVJvXv3Vo8ePfTzzz/r5Zdf1vDhwxUZGam5c+fqp59+0siRIxUUFKTnnntOt912W5EBKCMjQ7NmzdLhw4fVpk0bXX/99QoI+L8Dut9++63eeecdjR49WvXq1bPGc3NzNW7cOPXr10/du3c/Z40HDx4sto7t27fro48+UkZGhurXr6/+/furQYMG1uVn9levXj3NmjVL+/fvV6tWrXTjjTcqMDCwhPcK4D942QWwyfTp01WzZk3dcsst6t+/v+rXr6833nijyLkjR45U9+7dtWvXLuXm5urOO+/Ue++9p6SkJG3YsMFt7ltvvaUmTZpozpw5CgoK0vfff6927drpX//6V5nqjY6OVlxcnFavXq0jR44UO2/dunVq1KiR5syZo8DAQKWlpem+++7TqFGjJElOp1O1a9eWJEVERCgyMlKRkZEKDQ2V9H/ncCxfvlx9+vTRrl27tHPnTh08ePCs51rs2bNHffr0UVpamn777Tfdcccd6tGjh06cOGHN2bFjh5KSknT48GG3fXNzc5WUlKSvvvqqRDUWV8e4ceMUGxurzz//XEFBQZo3b54uvPBCvfvuu9ac0/2tXr1a1157rfbs2aOsrCwNHjxYgwYNKtF9AfgdA6DcnTp1ytSvX988+OCD1lhiYqKJiIgwv//+u9vcRYsWGUnmv//9rzXmcrnMTTfdZCSZp59+2hrfsGGDCQwMNI888ojbGu+++66RZD777LNia0pLSzOSzAMPPFDsnDFjxritU9Q+/fv3N+3bty+073fffWf991dffWUkmfnz5xeat2bNGiPJNGvWzBw+fNgaz8rKMlu2bDGSzOzZswvNb968udv8DRs2mICAAPP4449bY7NnzzaSzKZNm9yu88iRI0aSefbZZ0tUY1F1LF261EgyEydOtMYKCgpMfHy8qVKlitm1a5dbvS1btjTp6enW3FdffdVIMhs2bCh0fYC/48gHYIOlS5fqwIEDuueee6yxu+66Szk5OVqwYIHb3JkzZ6p27dpu54MEBQXpzjvvLLTuf//7XwUGBioxMdFtPD4+XlFRUZo5c2aZ6j79f/45OTnFzjl58qTS09N16NAht/HiXk4qzg033KC6deta2+Hh4WedP2DAALf5nTp1UlxcnN58802Prre03nrrLYWGhuqhhx6yxhwOh5566imdPHlS77zzjtv8m266yTq6IknXX3+9JFkv8QB/JZzzAdjgjTfeUEREhF577TW38bp16+qNN97QbbfdZo398MMPatq0qYKC3H89mzVrVmjdzZs3KywsTOPHj5ckGWOsH0llfjtqdna2JKlGjRrFzhk7dqz69++vxo0bq3v37urevbt69+7t8UmqzZs3L/P8iy++WMuWLdPx48et4FRefvjhBzVp0kTVqlVzG4+JiZHT6dQPP/zgNn7RRRe5bdetW1cBAQE6cOBAudYJVEYc+QDK2YEDB7R06VINHTrUOpfg9M+IESP02Wefuf2hcjgcys/PL7TOqVOnily/atWqqlOnjurUqaO6devqvPPO0/nnn6/Ro0eX+d00mzZtUmBg4FlPXu3WrZv27Nmj5ORkNWrUSMnJyWrbtq2GDx/u0XXVrFnTo/lF3UanxxwOhyQpODhYUuHbzhufseJwOFRQUFBo3BijgoICq4bT/hxSHA6HHA5Hsfcr4M848gGUszfffFNhYWGaMmVKoaMZ0h/vrpg+fbqeeeYZSVKLFi20ePFi5eXlyel0WvO2bdtWaN82bdpo7ty5uv/++1W1alWv1r1z506tXLlSffr0OeuRD+mP4BAfH6/4+HhJ0qhRo/Tiiy/qoYce0sUXX2y9o+P0ERlvKOrDwDZv3qwLLrhAISEhkv541470x7uBzrWvpzVefPHFev/995WTk6Pq1atb49u2bZPL5VKLFi1K1gjwF8SRD6AcGWOUnJysq666qsjgIUlXX3213n77bblcLknS8OHDlZWVpeeff96a8/vvv+utt94qtO+oUaN06tQpPfjgg4X+D3rPnj2l/jjwrVu3qn///qpVq5ZbHUVJSUnRyZMn3cYiIyMl/d+Rh6ioKEl/fJCXtyxevFh79+61tpcvX65Vq1a5He1p2bKlatWq5faBbseOHdP//ve/Qut5WuNdd92lEydOaMKECdaYy+XS448/rpCQEA0ePNjjnoC/Co58AOVoxYoV+umnn/TYY48VO6d379569dVXtWjRIt1444268sorNXHiRD3xxBNasWKFmjRpom+++UaJiYl677333D7HolWrVlq4cKGGDRum5cuX67LLLlNwcLB27NihI0eOKDk5+Zw1rlmzRgkJCTLGKDs7W1u3btXGjRvVr18/JSUlqVGjRmfdf9WqVRo5cqTatGmj6Oho7du3T6mpqXryyScVExMj6Y8w0qtXL02YMEHbt29XaGio9RkapTVq1CgNGjRIf/vb35Sbm6uUlBTdeOONbh8GFhoaqqefflp33323Dh06pCZNmujbb7/V5MmTtXDhQrf1PK3x8ssv19SpU/Xwww/riy++UGxsrNauXav9+/dr3rx5bp/1AcCdw3jzOCgAN6tWrdLXX3+tIUOGuL0z40zHjx/XtGnT1LJlS/Xq1csa37Fjh1atWiWn06k+ffooMzNTF110kaZNm6a7777bbY28vDytWrVKP/74o0JCQtSsWTN17tzZLaj82bFjx/Tqq69a20FBQQoLC1OjRo3UoUMHRUREFLtP+/bt3T7oLD09XWvXrlVaWprq1q2ryy+/3DqScGaNn3zyifbs2SOXy6WuXbvqkksu0f79+zV79mzdcMMNatKkids+GRkZevPNN9WvXz/rhM0z50dGRmrx4sXWh4x17dq1yF63bdum1atXKzQ0VP369VO1atX0yiuv6IorrlCHDh3OWWNRdZz266+/asWKFdaHjMXFxbm9THW2/p5//nm1b99e3bt3L7JuwF8RPgAfMX/+fN10001as2ZNsX9kAcAXED6ASmj9+vXq3Lmz9Y6Jo0eP6u9//7uOHTumnTt3nvWIBgBUdpzzAVRCy5Yt0x133KFLL71ULpdLK1asUGBgoBYtWkTwAODzOPIBVFK7d+/Wl19+qfT0dF1wwQXq0aNHoc+KAABfRPgAAAC24vgtAACwFeEDAADYqlKecFpQUKADBw6oevXqhb4fAQAAVE7GGOXk5KhevXpnPTm+UoaPAwcOKDo6uqLLAAAApZCWlnbWT/mtlOHj9Jc0paWlKTw8vMzruVwuffLJJ+rZs6f1XRP+xt979Pf+JHr0B/7en0SP/qA8+8vOzlZ0dLTbly0WpVKGj9MvtYSHh3stfISEhCg8PNwvH0iS//fo7/1J9OgP/L0/iR79gR39neuUCU44BQAAtiJ8AAAAWxE+AACArQgfAADAVoQPAABgK8IHAACwFeEDAADYivABAABsRfgAAAC2InwAAABbET4AAICtCB8AAMBWhA8AAGArwgcAALBVUEUXAHhb4zFLSr3vz1P6eLESAEBROPIBAABsRfgAAAC2InwAAABbET4AAICtCB8AAMBWhA8AAGArwgcAALAV4QMAANiK8AEAAGzlUfjo0aOHGjduXOjnzjvvdJv36aefqmfPnrrooovUt29fff31114tGgAA+C6PPl59xowZcrlc1vYvv/yibt26qWPHjtbYhg0bdPXVV+uJJ57Qv//9b7311lu64oortGnTJsXExHivcgAA4JM8Ch/169d3237nnXcUGhqq+Ph4a2zy5Mm64oorNG7cOElS27ZttXz5ciUlJenVV1/1QskAAMCXlfqcD2OM3nzzTd18882qXr26Nb569Wr16tXLbe7VV1+tVatWlbpIAADgP0r9rbaffvqp9uzZo1mzZlljOTk5ysrKUmRkpNvcyMhI7d+/v9i18vLylJeXZ21nZ2dLklwul9vLPKV1eg1vrFVZ+XuPnvTnDDRlvp6K4O/3oeT/Pfp7fxI9+oPy7K+kazqMMaV6po6Pj9fWrVu1efNmaywrK0s1a9bUnDlzNGjQIGt86tSpeuyxx5Sbm1vkWomJiRo/fnyh8VmzZikkJKQ05QEAAJvl5uYqPj5eWVlZCg8PL3ZeqY58HD16VB988IGeeeYZt/Hq1avL6XQqIyPDbTw9PV1169Ytdr2xY8dq9OjR1nZ2draio6PVs2fPsxZfUi6XS6mpqYqLi1NwcHCZ16uM/L1HT/qLTVxW6uvZmtjr3JPKib/fh5L/9+jv/Un06A/Ks7/Tr1ycS6nCx7vvvitJuu2229zGAwIC1L59e33xxRe69957rfHPP//c7R0xf+Z0OuV0OguNBwcHe/WG8fZ6lZG/91iS/vLyHWVav6L5+30o+X+P/t6fRI/+oDz6K+l6pTrhNDk5WQMGDFBEREShy+699169//77+uyzzyRJH3zwgdasWaN77rmnNFcFAAD8jMdHPjZu3Khvv/1WL774YpGX33rrrdqzZ4+uueYaBQYGyuFw6KWXXlKPHj3KXCwAAPB9HoePmJgY/fTTT2rcuHGxc5588kk9+uijysjIUN26dRUUVOo31QAAAD/jcSqoXr262+d6FKdKlSqKiooqVVEAAMB/8cVyAADAVoQPAABgK8IHAACwFeEDAADYivABAABsRfgAAAC2InwAAABbET4AAICtCB8AAMBWhA8AAGArwgcAALAV4QMAANiK8AEAAGxF+AAAALYifAAAAFsRPgAAgK0IHwAAwFZBFV0A8GeNxywpNOYMNHqmkxSbuEx5+Y4KqOrciqq7pH6e0seLlQBA5caRDwAAYCvCBwAAsBXhAwAA2IrwAQAAbEX4AAAAtiJ8AAAAWxE+AACArQgfAADAVoQPAABgK8IHAACwFeEDAADYivABAABsRfgAAAC24lttgUqg8Zglpf7mXr4RF4Cv4cgHAACwFeEDAADYivABAABsRfgAAAC2KtUJp0uXLtXSpUvlcDh0yy23qEuXLm6X79+/X6+99pr27t2rmJgY3XvvvapVq5ZXCgYAAL7NoyMfxhgNHjxYw4YNU4MGDdSmTRs98cQTWrZsmTVn3759ateunTZt2qRLLrlEy5cvV4cOHfTbb795vXgAAOB7PDryMWPGDM2dO1ebNm3SxRdfLEkaNmyYMjIyrDkTJkxQVFSUFi5cqMDAQN1xxx268MIL9cILL2jixInerR4AAPgcj458vPHGG+rTp48VPCTJ4XCoTp061vbSpUt1ww03KDAwUJJUrVo19evXT0uWLPFSyQAAwJd5dOTj22+/1bhx4zRjxgytXLlS5513nm644QZ17txZkvT777/r119/VcOGDd32a9SokebMmVPsunl5ecrLy7O2s7OzJUkul0sul8uTEot0eg1vrFVZ+VOPzkBTeCzAuP1bXspy+xVVt0f7l7JHX7rP/elxWhR/70+iR39Qnv2VdE2HMaZEz3TGGAUEBKhhw4bq0KGDrrnmGm3fvl0vv/yyXn/9dQ0ZMkRHjx5VrVq1NH/+fA0YMMDa96WXXtLDDz+sEydOFLl2YmKixo8fX2h81qxZCgkJKVEjAACgYuXm5io+Pl5ZWVkKDw8vdl6Jj3w4HA6Fh4erdu3aWrBggRyOPz7+uaCgQE899ZSGDBmisLAwBQQEFDq5NCMjQzVr1ix27bFjx2r06NHWdnZ2tqKjo9WzZ8+zFl9SLpdLqampiouLU3BwcJnXq4z8qcfYxGWFxpwBRhM7FOjJrwOUV1Dyjx731NbEXqXet6i6PVHaHstSs9386XFaFH/vT6JHf1Ce/Z1+5eJcPHrZpXXr1qpfv74VPCSpWbNm+vXXX2WMUXBwsJo3b66tW7e67bdlyxa1bNmy2HWdTqecTmeh8eDgYK/eMN5erzLyhx7P9r0meQUOj773xFNlue28VZenPfri/e0Pj9Oz8ff+JHr0B+XRX0nX8+iE0yFDhmjVqlXWu1vy8/O1cOFCde7c2Qokt956q+bOnasDBw5Ikr7//nt99NFHuu222zy5KgAA4Kc8OvIxdOhQrV27Vs2bN9cll1yiHTt2KCAgQB9++KE1Z/To0fr888/VunVrtWvXThs2bNDAgQN1++23e714AADgezwKHwEBAfrf//6n77//Xjt37lRUVJTatWvndpjF6XRqyZIl+uabb7Rv3z4lJSUpNjbW64UDAADfVKqPV2/evLmaN29+1jnt27dX+/btS1UUAADwX3yxHAAAsBXhAwAA2IrwAQAAbEX4AAAAtiJ8AAAAWxE+AACArQgfAADAVoQPAABgK8IHAACwFeEDAADYivABAABsRfgAAAC2InwAAABbET4AAICtCB8AAMBWhA8AAGArwgcAALAV4QMAANiK8AEAAGxF+AAAALYifAAAAFsRPgAAgK0IHwAAwFaEDwAAYCvCBwAAsBXhAwAA2IrwAQAAbEX4AAAAtiJ8AAAAWxE+AACArQgfAADAVoQPAABgK8IHAACwFeEDAADYivABAABsRfgAAAC2CvJk8pdffqkvvvjCbczpdOqee+5xGzt16pQ++eQT7d27VzExMerRo4ccDkfZqwUAAD7Po/DxySef6JVXXtGgQYOssWrVqrnNOXbsmK666iodOXJEXbt21aRJk9SyZUstWrRIwcHB3qkaAAD4LI/ChyQ1atRIU6dOLfbyKVOmaP/+/dq8ebMiIiKUlpamFi1a6PXXX9d9991XlloBAIAf8Picj8zMTCUnJ+vdd9/Vzp07C10+b948DRo0SBEREZKk6Oho9e3bV/PmzSt7tQAAwOd5fOQjNzdXn332mTIzMzVs2DDdd999ev755yVJJ0+e1I8//qhmzZq57dOsWTOlpqYWu2ZeXp7y8vKs7ezsbEmSy+WSy+XytMRCTq/hjbUqK3/q0RloCo8FGLd/y0tZbr+i6vZo/1L26Ev3uT89Tovi7/1J9OgPyrO/kq7pMMaU+Jlu8+bNatasmapUqSJJWrVqlXr06KEFCxbo+uuv19GjR1WrVi3Nnz9fAwYMsPZ76aWX9PDDD+vEiRNFrpuYmKjx48cXGp81a5ZCQkJKWh4AAKhAubm5io+PV1ZWlsLDw4ud59GRj1atWrltX3HFFWrbtq1SU1N1/fXXW0EhJyfHbV52dvZZQ8TYsWM1evRot/nR0dHq2bPnWYsvKZfLpdTUVMXFxfntSa/+1GNs4rJCY84Ao4kdCvTk1wHKKyi/d05tTexV6n2LqtsTpe2xLDXbzZ8ep0Xx9/4kevQH5dnf6VcuzsXjl13+rEqVKsrKypL0x9tuGzZsqN27d7vN2b17t2JiYopdw+l0yul0FhoPDg726g3j7fUqI3/oMS+/+D+8eQWOs15eVmW57bxVl6c9+uL97Q+P07Px9/4kevQH5dFfSdfz6ITT77//3m1727Zt+uabb9S1a1drrH///po/f751DkdmZqZSUlJ03XXXeXJVAADAT3l05GPkyJEKDw9X27ZtlZ6errfeeku9evXS8OHDrTlPPPGEFi9erCuvvFJxcXFauHCh6tevr5EjR3q9eAAA4Hs8OvKRmpqqESNGKD8/X/Xr19eHH36oRYsWWSegStJ5552nTZs26dZbb9WJEyc0cuRIrVu3TmFhYV4vHgAA+B6Pjnw4HA717t1bvXv3Puu8GjVq8IFiAACgSHyxHAAAsBXhAwAA2IrwAQAAbEX4AAAAtiJ8AAAAWxE+AACArQgfAADAVoQPAABgqzJ/sRwA39V4zJJS7/vzlD5erATAXwlHPgAAgK0IHwAAwFaEDwAAYCvCBwAAsBXhAwAA2IrwAQAAbEX4AAAAtiJ8AAAAWxE+AACArQgfAADAVoQPAABgK8IHAACwFeEDAADYivABAABsRfgAAAC2InwAAABbET4AAICtCB8AAMBWhA8AAGArwgcAALAV4QMAANiK8AEAAGxF+AAAALYKqugCgMqk8ZglFV0CAPg9jnwAAABbET4AAICtCB8AAMBWpQ4fJ06c0Pr167Vr164iLz9y5Ig2bdqkzMzM0l4FAADwQ6UOHw8++KC6dOmisWPHuo0bY3TfffepQYMGuvnmmxUZGamnnnqqzIUCAAD/UKp3u7z//vtat26devToUeiy1157Te+88442btyoFi1a6PPPP9eVV16ptm3b6rrrritrvQAAwMd5fORj7969uv/++/Xuu+/K6XQWunz69OkaMGCAWrRoIUnq2rWrevTooeTk5LJXCwAAfJ5H4ePUqVO65ZZbNHbsWCtcnCk/P1+bN29Wx44d3cY7d+6sTZs2la1SAADgFzx62WXcuHGqUaOGRo4cWeTlOTk5crlcqlWrltt47dq1lZGRUey6eXl5ysvLs7azs7MlSS6XSy6Xy5MSi3R6DW+sVVn5U4/OQFN4LMC4/euPSttjWe7zom7r8rxef3qcFsXf+5Po0R+UZ38lXdNhjCnRs8/GjRt12WWXafbs2YqMjJQkJSQkqEqVKpo8ebLatm2rU6dOKSwsTDNmzNDtt99u7fvvf/9bkyZNskLFnyUmJmr8+PGFxmfNmqWQkJASNQIAACpWbm6u4uPjlZWVpfDw8GLnlfjIx7Fjx9S6dWtNmTLFGtu5c6cCAgI0atQoffDBB4qKilLt2rV14MABt30PHDigRo0aFbv22LFjNXr0aGs7Oztb0dHR6tmz51mLLymXy6XU1FTFxcUpODi4zOtVRv7UY2ziskJjzgCjiR0K9OTXAcorcFRAVeWvtD1uTexV6uss6rYuz+v1p8dpUfy9P4ke/UF59lfcQYY/K3H4uPzyy7V+/Xq3sb59+6pq1apasGCBNdajRw+lpKTo0UcflfTHeSBLlixR3759i13b6XQWefJqcHCwV28Yb69XGflDj3n5xf/hzStwnPVyf+Bpj2W5v8tyW5blev3hcXo2/t6fRI/+oDz6K+l6Xv+E0yeffFKbNm3Svffeq48++ki33367MjMzlZCQ4O2rAgAAPqhM32rbrFkzValSxW0sNjZWa9euVVJSkqZMmaKYmBitW7dODRo0KFOh8C18OywAoDhlCh/PPfdckeNt2rTRzJkzy7I0AADwU3yxHAAAsBXhAwAA2IrwAQAAbFWmcz4A/HWV5qRiZ6DRM53KoRgAPoUjHwAAwFaEDwAAYCvCBwAAsBXhAwAA2IrwAQAAbEX4AAAAtiJ8AAAAWxE+AACArQgfAADAVoQPAABgK8IHAACwFeEDAADYivABAABsRfgAAAC2InwAAABbET4AAICtCB8AAMBWhA8AAGArwgcAALAV4QMAANiK8AEAAGxF+AAAALYifAAAAFsFVXQBAMqm8ZglFV0CAHiEIx8AAMBWhA8AAGArwgcAALAV4QMAANiK8AEAAGxF+AAAALYifAAAAFsRPgAAgK0IHwAAwFYehY9ff/1VDzzwgBo3bqywsDC1a9dO7777bqF5M2bM0N/+9jc5nU7FxsYqJSXFawUDAADf5lH4mDZtmlq1aqW1a9fq119/1T333KPBgwdr2bJl1pyPP/5Yw4cP1xNPPKGDBw9q2LBhuuGGG/TNN994vXgAAOB7PAofEyZM0PDhw1W/fn1Vr15dI0aMUP369bV+/XprznPPPad+/fpp8ODBioiI0OjRo9WmTRtNnTrV27UDAAAfVOpzPo4fP663335bv/32m/r27StJMsZo/fr1uuKKK9zm9ujRQ1988UWZCgUAAP7B42+13bp1q1q2bClJqlq1qqZPn6727dtLknJycnT8+HHVrVvXbZ/zzjtPBw8eLHbNvLw85eXlWdvZ2dmSJJfLJZfL5WmJhZxewxtrVVaVrUdnoPHuegHG7V9/9FfqsbI8Tr2tsv0elgd69H3l2V9J13QYY0r1TJeZmal58+bp/vvv14IFC3TttdcqOztbNWrU0KxZs3TLLbdYc5OSkvTUU0/p2LFjRa6VmJio8ePHFxqfNWuWQkJCSlMeAACwWW5uruLj45WVlaXw8PBi55U6fJx23XXX6eTJk1q6dKmMMapevbomT56sf/7zn9acMWPGaP78+dq9e3eRaxR15CM6Olrp6elnLb6kXC6XUlNTFRcXp7aTPi31OlsTe5W5lvJyZo/BwcEVXY5iE5ede5IHnAFGEzsU6MmvA5RX4PDq2pXFX6nHyvI49bbK9ntYHujR95Vnf9nZ2apTp845w4fHL7v8mcvlksPxxxOlw+HQJZdcopUrV7qFj08//VRdunQpdg2n0ymn01loPDg42Ks3THBwsPLyS/+k7gsPQm/fZqVVltv5rOsWOMpt7crir9BjZXmclhd/70+iR39QHv2VdD2PTjgdOHCgPv/8c+Xk5Ojw4cN64YUX9PHHH+uOO+6w5iQkJGjx4sWaMWOGjh49qhdeeEGbNm3SqFGjPLkqAADgpzw68vHPf/5TEydO1IYNGxQcHKwWLVroww8/tN7tIklXX321kpOTNXHiRI0YMUIxMTF67733rJNSAQDAX5tH4aNbt25uHyhWnMGDB2vw4MGlLgqVQ+MxSyq6BACAH+K7XQAAgK0IHwAAwFaEDwAAYKsyv9UWAOxUlnORfp7Sx4uVACgtjnwAAABbET4AAICtCB8AAMBWhA8AAGArwgcAALAV4QMAANiK8AEAAGxF+AAAALYifAAAAFsRPgAAgK0IHwAAwFaEDwAAYCvCBwAAsBXhAwAA2IrwAQAAbEX4AAAAtiJ8AAAAWxE+AACArQgfAADAVoQPAABgK8IHAACwFeEDAADYivABAABsRfgAAAC2InwAAABbET4AAICtCB8AAMBWhA8AAGArwgcAALAV4QMAANiK8AEAAGxF+AAAALYifAAAAFsFeTL5+PHj+t///qfVq1fL5XKpY8eOGjlypGrUqOE2b8uWLXrhhRe0d+9excTE6JFHHlGTJk28WjgA3xWbuEx5+Q7br7fxmCWl3vfnKX28WAnw1+bRkY9u3brpxx9/1M0336w77rhDKSkp6tKli44dO2bN2b59u7p06aKgoCCNGjVKv/32my655BLt37/f68UDAADf49GRj1WrVik8PNza7tq1q84//3x99NFHGjhwoCRp4sSJatWqlV5//XVJ0jXXXKOYmBglJSXp+eef92LpAADAF3l05OPM4CFJoaGhCgwM1IkTJ6yx5cuXq1+/ftZ2YGCg+vbtq9TU1DKWCgAA/IFHRz7+7LnnnlOVKlV01VVXSfrjnJD09HTVr1/fbV79+vW1b9++YtfJy8tTXl6etZ2dnS1JcrlccrlcZSnRWuf0v85AU+Z1KqMze/SWstxW3uYMMG7/+iN6rNxK8rtVHr+HlQ09+r7y7K+kazqMMaV6Fvjggw80cOBAvfnmm7r99tslSZmZmYqIiNC8efOsl2Ek6T//+Y8effRR/f7770WulZiYqPHjxxcanzVrlkJCQkpTHgAAsFlubq7i4+OVlZVV6NWSM5XqyMeSJUt0yy23aOrUqVbwkKTq1asrODhYGRkZbvMzMjJUq1atYtcbO3asRo8ebW1nZ2crOjpaPXv2PGvxJeVyuZSamqq4uDi1nfRpqdfZmtirzLWUlzN7DA4O9sqasYnLvLKONzgDjCZ2KNCTXwcor8D+d0nYgR4rt5L8/pfH72FlQ4++rzz7O/3Kxbl4HD6WLl2qG2+8Uc8++6zuv/9+t8sCAwPVsmVLffPNN27jX375pdq2bVvsmk6nU06ns9B4cHCwV2+Y4ODgMr29zxcehN68zSrirZDnklfgqJR1eRM9Vk6e/F55+7mrMqJH31ce/ZV0PY9OOP3444+t4DFy5Mgi5wwbNkzz58/Xjh07JEnr1q3TihUrNGzYME+uCgAA+CmPjnwMGjRIgYGBmjt3rubOnWuNDxs2zAoX99xzjzZv3qw2bdqoadOm2r17txISEnTDDTd4t3IAAOCTPAofH330kQoKCgqNN2zY0PrvgIAAvfbaaxo/frzS0tJ0wQUXqE6dOmWvFAAA+AWPwkeXLl1KPDcyMlKRkZEeFwQAAPwbXywHAABsRfgAAAC2InwAAABbET4AAICtCB8AAMBWhA8AAGArwgcAALAV4QMAANiK8AEAAGxF+AAAALYifAAAAFsRPgAAgK08+mI5APirajxmyTnnOAONnukkxSYuU16+wxr/eUqf8iwN8Dkc+QAAALYifAAAAFsRPgAAgK0IHwAAwFaccAoAlVhJTnQtDie6orLiyAcAALAV4QMAANiK8AEAAGxF+AAAALbihFMPVNSJX3yyIuDbyvLcAfgjjnwAAABbET4AAICtCB8AAMBWhA8AAGArwgcAALAV4QMAANiK8AEAAGxF+AAAALYifAAAAFsRPgAAgK0IHwAAwFaEDwAAYCvCBwAAsJXH32p78uRJLViwQBs3blR8fLzatWtXaE5OTo5mz56tvXv3KiYmRjfffLOqVq3qlYIBAIBv8+jIR0pKipo0aaJFixYpKSlJ27dvLzQnPT1d7dq1U3JysiQpKSlJl112mY4fP+6digEAgE/zKHw0adJEGzdu1Jw5c4qd869//UsOh0OrVq3SpEmTtHr1au3du1cvv/xymYsFAAC+z6Pw0aJFC5133nlnnbNw4UINHDhQ1apVkyTVqlVL/fr108KFC0tdJAAA8B8en/NxNnl5edq7d6+aNm3qNt60aVMtWrTorPvl5eVZ29nZ2ZIkl8sll8tV5rpOr+FyueQMNGVeryw1lEZJanYGGLd/7bpeuxTXnz+hR99X2frzxvNncWuWx9qVhb/3WJ79lXRNhzGmVL8lDodDM2fO1G233WaNHT16VLVq1dL8+fM1YMAAa/yll17Sww8/rBMnThS5VmJiosaPH19ofNasWQoJCSlNeQAAwGa5ubmKj49XVlaWwsPDi53n1SMfoaGhcjgcyszMdBvPzMxU9erVi91v7NixGj16tLWdnZ2t6Oho9ezZ86zFl5TL5VJqaqri4uLUdtKnZV6vMnIGGE3sUKAnvw5QXoHDGt+a2KvUa8YmLvNGaV5RXH/+hB59X2Xrryy//8U58/k0ODjY6+tXBv7eY3n2d/qVi3PxavioUqWKmjZtqp07d7qN79ixQ82bNy92P6fTKafTWWg8ODjYqzdMcHCw8vIr/gmhPOUVONx6LMvtVxlvqz/354/o0fdVlv7K8w+nt5+fKyN/77E8+ivpel7/kLGBAwdq3rx5ysrKkiTt379fixcv1sCBA719VQAAwAd5dORj165deu2116zt2bNn69tvv1Xnzp2tcDFmzBh98skn6tixo7p166bU1FR16tRJ//jHP7xbOQAA8EkehQ+n06nIyEhJ0rPPPmuN16hRw/rv8PBwrVu3TkuXLtW+ffs0cOBA9ezZUwEBfJI7APiKxmOWFDnuDDR6ptMf54QV99LSz1P6lGdp8AMehY+GDRsqISHhnPOCg4PVv3//UhcFAAD8F4cjAACArQgfAADAVoQPAABgK69+zgcAAMWdrFoSnKz618CRDwAAYCvCBwAAsBXhAwAA2IrwAQAAbMUJpwDgp8py4idQnjjyAQAAbEX4AAAAtiJ8AAAAWxE+AACArQgfAADAVoQPAABgK8IHAACwFeEDAADYig8ZAwD85Z35gWzOQKNnOkmxicuUl+845758E6/nOPIBAABsRfgAAAC2InwAAABbET4AAICtOOEUAFBplOWbeDnx03dw5AMAANiK8AEAAGxF+AAAALYifAAAAFtxwqmfK8vJWwAAlAeOfAAAAFsRPgAAgK0IHwAAwFaEDwAAYCvCBwAAsBXhAwAA2IrwAQAAbEX4AAAAtiq3DxnbvHmz9u7dq5iYGDVr1qy8rgYAAPgYr4ePkydP6qabbtKaNWvUpk0bffXVVxo4cKCmT58uh8Ph7asDAKBCleWTpH+e0sf263UGGj3TqdRX6xVeDx9Tp07V2rVr9d1336lBgwbatm2bOnTooCuuuEK33367t68OAAD4GK+f8zFz5kwNGjRIDRo0kCS1aNFCvXv31syZM719VQAAwAd59cjHqVOn9P3332vkyJFu461atdKrr75a7H55eXnKy8uztrOysiRJv/32m1wuV5nrcrlcys3NVUZGhoJOHS/zepVRUIFRbm6BglwByi/wv5e3/L0/iR79gb/3J1XuHjMyMkq975l/G+zs0Vs1e7Tf/+8vIyNDwcHBpb7+ouTk5EiSjDFnn2i86OjRo0aSmTdvntv4f/7zH+N0Oovd76mnnjKS+OGHH3744YcfP/hJS0s7a17w6pEPp9MpScrNzXUbP3bsmKpWrVrsfmPHjtXo0aOt7YKCAv3222+qXbu2V05Szc7OVnR0tNLS0hQeHl7m9Sojf+/R3/uT6NEf+Ht/Ej36g/LszxijnJwc1atX76zzvBo+qlWrpsjISO3bt89tfN++fWrSpEmx+zmdTiu4nFazZk1vliZJCg8P98sH0pn8vUd/70+iR3/g7/1J9OgPyqu/GjVqnHOO10847d27t95//30VFBRIkk6cOKGUlBT17t3b21cFAAB8kNfDx7hx4/TLL7/oxhtv1BtvvKE+ffooKCjI7WUVAADw1+X18NG4cWNt3LhRzZo106pVq9StWzd99dVXql27trevqsScTqeeeuqpQi/t+BN/79Hf+5Po0R/4e38SPfqDytCfw5hzvR8GAADAe/hiOQAAYCvCBwAAsBXhAwAA2MrrXyxX2Rw6dEgbNmxQaGiounbt6vMnELlcLi1fvlwnT55U//79i5yTnZ2ttWvXSpIuu+wyn3qf+smTJ7Vp0yalp6erefPmxX4+zI8//qitW7fq/PPPV+fOnRUQ4Ds5Ojc3V19//bWOHz+uiy++WI0aNSo0p6CgQOvXr9fhw4fVsmVLNW3atAIqLbt169Zp79696tWrlyIiItwuy8rK0tq1a+VwONS1a1dVr169gqr0zIEDB/TZZ58VGu/Xr59CQ0Pdxnbt2qVt27YpMjJSnTp18qnHqfTHV1xs2LBB1atX1yWXXKKgIPc/Gfn5+dqwYYMOHz6sVq1anfXznCqTX375RZ9//nmRl1111VWqU6eOtZ2Zmam1a9cqMDBQXbt2VVhYmF1lesXGjRuVlpamWrVqqWPHjkV+4OcPP/yg7du3KyoqSp06dbLnG+jL/qHqldfbb79tQkJCzOWXX26aNWtmGjVqZHbu3FnRZZXahAkTTHR0tGncuLE5//zzi5yzcuVKExERYTp06GA6duxoIiIizMqVK+0ttJRmzpxpLrjgAtOpUydzzTXXmLCwMHPbbbeZU6dOuc179NFHTWhoqImLizP16tUzl1xyicnMzKygqj3z2muvmUaNGpm///3vJi4uzlSrVs3ceeedJj8/35rz22+/mY4dO5r69eubq666yoSEhJjHH3+8Aqsune+++86Eh4cbSearr75yu2z58uWmZs2aplOnTqZ9+/amVq1aZvXq1RVUqWdSUlJMQECAGTRokNvP4cOH3eY99NBD1uM0KirKdOnSxWRlZVVQ1Z5LSkoyoaGhpnv37qZXr16mc+fO5uDBg9bl6enppn379qZBgwbW43TcuHEVWHHJrV+/vtD916JFC+NwOMzPP/9szVu2bJmpUaOG6dy5s2nbtq2pXbu2+fzzzyuw8pI7fPiwadeunalXr5659tprTYsWLcz5559vvvjiC7d5o0aNMmFhYSYuLs5ERkaabt26mezs7HKvz2/DR1pamnE6nWbatGnGGGNOnTplevbsabp27VrBlZXe1KlTzYEDB8yzzz5bZPg4ceKEqVevnnnggQessfvuu8/Ur1/f5OXl2Vhp6cyZM8ccOHDA2t61a5cJDQ01r7zyijWWmppqHA6HWbt2rTHGmMzMTNO0aVNz//33215vaSxYsMDk5ORY2xs3bjSSzNKlS62xf/zjH6ZZs2bWE8DKlSuNJLNq1Srb6y2t48ePm+bNm5tnnnmmUPjIzc01kZGR5qGHHrLG/vGPf5jo6Ghz8uTJiijXIykpKWf9ripjjPnoo49MQECAWb9+vTHmj0B5wQUXmFGjRtlRYpnNnTvXBAYGmhUrVlhj3333nfnpp5+s7eHDh5sWLVpYj+fU1FQjyaxZs8bucr3isssuM1dddZW1ffz4cVO3bl3z6KOPWmPDhw83jRs3Ni6XqyJK9Mhjjz1m6tWr5/Z807dvX9OlSxdrOyUlxQQGBlq/n+np6aZhw4YmISGh3Ovz2/DxwgsvmPDwcLc/ukuXLjWS3H6BfFFx4eOjjz4yktyS+549e4wk8/HHH9tZotdceumlZsSIEdb2kCFDzKWXXuo2Z/LkySYiIsIUFBTYXV6ZZWdnm4CAADN37lxjjDH5+fkmPDzcJCUluc1r166dufPOOyuixFIZOnSouf/++82WLVsKhY+UlJRCXzz1ww8/GElm+fLlFVGuR1JSUkyVKlXMJ598YpYuXWr27t1baM6tt95qunXr5jY2YcIEU6dOHbvKLJPWrVub+Pj4Yi93uVwmNDTUvPjii27jrVq1MnfffXd5l+d1O3bsMJKs30NjjPnggw+Mw+Ewv/76qzW2fft2n/kfgYceesi0atXKbWzkyJGmQ4cO1vagQYPMlVde6TZn3LhxJjIystzr860XID2wZcsWXXTRRapSpYo11rJlS0nS1q1bK6qscrVlyxZVr17d7RyCCy64QKGhodqyZUsFVlY6R44c0ebNmxUbG2uNbdmyxW1b+uN+PXr0qPbv3293iaVy4MABzZkzR6+99pp69+6ta6+9Vtdff70kae/evcrOzi6yR1+5D2fPnq0NGzbomWeeKfLyLVu2KCIiQg0aNLDGYmJiVLVqVZ/p0eFw6F//+peeeeYZxcTE6M4771R+fr51eXGP0/T0dB08eNDucj2SlZWl7777Tj179tTu3bv14Ycf6uuvv7a+MkOSfvrpJx0/ftynH6dnSk5OVp06dXTddddZY1u2bFGdOnUUGRlpjTVv3lzBwcE+0ePo0aMVFBSkoUOH6q233tKTTz6plJQUPf/889ac4h6nBw8eVHp6ernW57cnnGZlZalWrVpuY6c/ZTUzM7MCKip/RfUs/dG3r/Wcn5+vwYMHq169eho+fLg1fq779cw/aJXVoUOHtHDhQqWnp+vHH3/UNddcY53Il5WVJUlF9ugL9+Hu3bv1z3/+U6mpqapWrVqRc4p7nNaqVcsnemzatKl27Nihxo0bS5K+++47denSRRdddJEefvhhSed+nJ75B62yOXLkiCRp8eLFGj9+vGJjY7Vx40add955WrJkiaKios76OP3222/tLrlMTp06pRkzZmjIkCFu/7Na3OM0IiLCJx6nNWvW1KWXXqpFixYpIyNDP/74o5o1a6bo6Ghrzrkep2eeeOttfhs+nE5nof/DOHbsmCQVebavP3A6nVaPZzp27JhP9VxQUKAhQ4Zo8+bNWr16tds7CIrq0dfu17Zt22rOnDmS/jgK16lTJ0VFRWno0KHWu7GK6tEX+nvggQfUvn177dixQzt27NAvv/wiSfrkk0/kcrl06aWX+vzjtHnz5m7brVu31oABA5SSkmKFD19+nJ6ub/fu3dq+fbuqVq2q3NxcderUSY888ohmzpzp84/TMy1evFiHDh3SnXfe6Tbu64/ThIQEffrpp9q+fbvCwsJkjNHgwYPVr18/68hNRT5O/fZll6ZNm2rfvn1uY3v37pUkn3k7mKeaNm2qo0ePKicnxxrLzs5WZmamz/RcUFCgoUOH6tNPP9XKlSt14YUXul1e3P0aFBSkhg0b2lmqV8TGxqpt27Zas2aNJKlRo0YKDAwsskdfuA/btm2rmjVrauHChVq4cKFWrFghSVq9erW++uorSX/chxkZGcrNzbX2y8zMVHZ2tk/0WJTw8HDriIFU/OO0SpUqlf7oXFRUlKpVq6Y+ffpYf4BCQkLUu3dvffPNN5L+eDnX4XD47OP0TMnJyerWrZuaNWvmNt60aVMdOXJEJ06csMbS09OVm5vrEz2uXr1avXv3tt4a7HA4dOONN2rr1q3WSyrFPU6rVq2qqKio8i2w3M8qqSDr1q0zkqyzzY0xJiEhwdSrV8/tbY2+qLgTTg8dOmSqVKli3nrrLWts+vTppkqVKubQoUN2llgq+fn5ZsiQISYqKsrs2LGjyDnTpk0zISEhJiMjwxqLi4szV199tV1lltrJkydNenq621h2drapW7eu21sUe/ToYfr27WttHzp0yDidTjN9+nTbavWWok44PXDggAkODjbvvPOONfbqq68ap9NZ6PapjM58R5Yxxvz++++madOm5vbbb7fGXnrpJRMWFmaOHj1qjV155ZVu92tlduONN5oBAwa4jfXp08f07NnT2u7evbu57rrrrO3T9+uZzz+V3f79+01gYKCZMWNGocvS0tJMUFCQmTNnjjX28ssvm2rVqrndr5VVr169TK9evdzGJk+ebKpVq2b9DXzhhRdMjRo1rI8qKCgoMN26dXO7X8uL34YPY4yJj483DRo0MC+88IJ5+OGHTVBQkNvZzL5m1apVZvbs2ebWW281NWrUMLNnzzazZ892e0/2hAkTTFhYmJk0aZKZNGmSCQsLMxMnTqzAqkvuwQcfNA6Hw4wfP97qbfbs2eazzz6z5pw4ccK0a9fOtGvXzrzyyivm1ltvNaGhoWbTpk0VV3gJnX77aUJCgklOTjbPPfeciY2NNX/729/cPiPi66+/NiEhIWbw4MHm5ZdfNq1btzYdO3b0ibdL/1lR4cOYP86oDw8PN5MnTzYTJ040ISEh5umnn66gKj0zdOhQM2jQIPPyyy+bqVOnmtatW5uGDRuaPXv2WHN+//1307p1a9OhQwfzyiuvmFtuucWEhYWZzZs3V2DlJbdz505Tq1Ytc9ddd5nk5GQzbNgwU7VqVbfPiNiwYYOpWrWqGTp0qHn55ZdNy5YtTefOnX3ibainTZ482dSsWdPk5uYWefljjz1matSoYZ5++mkzYcIEU61aNfPss8/aXGXprFq1ygQFBZnbbrvNJCcnmzFjxphq1aqZSZMmWXOOHz9uWrZsaTp16mT++9//mkGDBpnq1aubbdu2lXt9fv2ttgUFBZoxY4Y+++wzhYSEKD4+Xl26dKnoskptypQpRZ7M9fzzz6tevXrW9qJFi7R48WJJUt++fXXttdfaVWKZJCYmaseOHYXG27Vrp0ceecTaPn78uKZNm6YtW7bo/PPP14gRIxQTE2NnqaV27Ngxvf3229q0aZOqV6+udu3aadCgQW4nuknSzp07NX36dB0+fFitW7fW3XffrZCQkAqquvR++eUXJSQkaPLkyYUOVS9cuFBLliyRw+HQtddeq759+1ZQlZ5btGiRli9frvz8fMXGxuqOO+4odILtsWPHNG3aNG3dulWRkZEaMWJEoZcRK7O0tDS9/vrrSktLU6NGjXTHHXfoggsucJvz/fffKzk5WUeOHFGbNm109913F3uicWX0+OOPKyoqSvfff3+xcz744AMtXbpUAQEBuu6669S7d28bKyybH3/8UTNnztS+fftUp04d9enTR1dccYXbnJycHE2bNk3btm1TVFSU7rrrLlteVvLr8AEAACofvz3hFAAAVE6EDwAAYCvCBwAAsBXhAwAA2IrwAQAAbEX4AAAAtiJ8AAAAWxE+AACArQgfAADAVoQPAABgK8IHAACwFeEDAADY6v8BD0jyMZSmp9sAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['Age'].hist(bins = 30)\n",
    "plt.title('Age DIstribution')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "cc9b082b-6e84-4a0c-aa62-bc60524c3174",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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8KB69+6t6dOnX3UdgNKDwAPA46Wnp+vcuXMKDQ3V2bNnNXDgQI0fP15TpkyRdOlXqA8cOHDF7WfOnOn6b6fTqZEjR2rixIlatmyZJGnBggXy9vbW/v375e196V6O5cuXX3MdgNKDwAPAI+3Zs0eLFi1Senq65syZI4vFor///e9asmSJ/P399fzzz7tqK1WqpLvuuuuq+zt8+LB+++03paenq1KlSvr6669d68qVKyebzaYTJ04oPDxcktSvX79rrgNQehB4AHikffv2KTMzU+XLl1efPn306KOPqmrVqjp58qTCw8Pl5+dXqP3k5ubqgQce0KZNm9S2bVsFBwcrISHB7bb1oUOHaseOHbr11lvVqFEj3XPPPXrssccUHh5+1XUASg8CDwCP9OeLlv8sODhYSUlJhd7Pl19+qa1bt+r48eOqVKmSJGn+/PnaunWrqyYgIEAff/yx3nvvPe3YsUP//ve/1apVKx05ckQhISFXXQegdODBgwBKlXvvvVcnTpzQli1b3MYTExMLrD937pyqVavmCjuStGTJErea06dPS7p0+qpTp07697//rdTUVB06dOiq6wCUHszwAChV2rRpoyeffFK9evXSU089pdq1a2vt2rXq1q2bxowZk6++e/fumjBhgkaNGqW2bdtq7dq12rx5s1vNwoULtXz5cvXp00dVq1bVkiVL1LBhQ91+++16//33r7gOQOnBk5YBeJzhw4fr4Ycf1r333nvFmjVr1mj16tVyOp3q0aOHevfuLUnKy8vTww8/rMmTJ6tZs2aSpL1792rOnDmy2Wy6/fbb1bZtW7333ntasGCBa3/ff/+9li9frqSkJDVr1kzDhw9X5cqVr7kOQOlA4AEAAKbHNTwAAMD0CDwAAMD0CDwAAMD0CDwAAMD0CDwAAMD0CDwAAMD0CDwAAMD0CDwAAMD0CDwAAMD0CDwAAMD0CDwAAMD0CDwAAMD0/j80DKgR56u34AAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data = df , x = 'Pclass', hue = 'Survived')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "faed4b99-88b2-4724-a8d0-13998aacc19e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjwAAAGxCAYAAABmyWwBAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlcelbwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAJwRJREFUeJzt3X1U1HWix/EPDE+SAoUoCAhqmEZqafnU+pR103LzuiVqmGZ2dF2T2yk7LnbTHixOm1a3u+eap0xzMe2mtaZkW5lmD9f1+lShrZRK4gMiIgOKDDDM/cN1biNqgwz+hi/v1zlzzs53vjPzHfb88n1+v9/8JsDlcrkEAABgsECrFwAAANDYCB4AAGA8ggcAABiP4AEAAMYjeAAAgPEIHgAAYDyCBwAAGI/gAQAAxguyegH+oLa2VkeOHFGrVq0UEBBg9XIAAIAXXC6XysvL1a5dOwUGXnofDsEj6ciRI0pMTLR6GQAA4DIUFBQoISHhknMIHkmtWrWSdPYPFhERYfFqAACAN8rKypSYmOj+d/xSCB7JfRgrIiKC4AEAoInx5nQUTloGAADGI3gAAIDxCB4AAGA8zuEBAMAPOJ1OVVdXW70MvxIcHCybzeaT1yJ4AACwkMvlUmFhoUpLS61eil+KiopSbGxsg6+TR/AAAGChc7HTpk0bhYeHcwHcf3K5XKqoqFBRUZEkKS4urkGvR/AAAGARp9Ppjp3o6Girl+N3WrRoIUkqKipSmzZtGnR4i5OWAQCwyLlzdsLDwy1eif8697dp6PlNBA8AABbjMNbF+epvQ/AAAIA6amtr9fDDD2vv3r2N/l7Tpk3T7t27G/U9CB4AAPyUy+XSmjVrNHv2bM2cOVPZ2dmqrKy8Iu9dW1urxYsX6+jRo43+XkuWLFFBQUGjvgfBAwCAH3I6nRo+fLgyMjIUEhKihIQEbdiwQTfccIMOHTrU6O9vs9n0xhtv6Lrrrmv097oS+JYWAAB+aOPGjfrb3/6mvLw8paSkuMePHTumsLAwSdLx48eVmZmpF1980eNbXlOnTlVGRoZSU1N15swZzZgxQ7NmzdKnn36q3NxcjR49Wl9//bWuv/56/e53v/N438cff1x33XWXhgwZoi1btmjAgAGKi4vTvHnzLjl/6NChkqSvvvpK69atU3V1tXr16qWxY8cqMPD/9684nU4tW7ZMO3bsUHJystLT033+t7sQ9vAAAOCHiouLJUmRkZEe423btnWP2e12LV68WOXl5R5zFi9e7D5E5HA4tHjxYt12223avn27evToobZt28rhcOjZZ5/1eN7OnTv18ssvq3379nUOaf3afEl65plnNGbMGIWFhSk+Pl4LFizQ8OHD5XK53M956KGH9OSTTyo2NlaFhYXq06ePampqGvrn+lXs4bmCej2xzOol4J+2vzTB6iUAwCUNHTpUMTExGjRokCZPnqx+/frplltuUUhIyGW9XlpamhYsWOC+P378eM2bN0+7d+9WamqqJGn58uXq3bu3UlJS6kTIr83Pzc3V888/r71796pDhw6SpClTpqhjx4768MMPNXLkSO3YsUN/+ctftGPHDt14442SpA4dOmj69OmX9Znqgz08AAD4oZiYGO3atUt33323li5dqkGDBumaa65RRkaGHA5HvV9v+PDhHvevu+469erVS8uXL5d09iTllStXavz48Rd8/q/Nz8nJUXh4uF588UVNnTpVU6dO1WOPPSabzaZdu3ZJkr744gtde+217tiRpDFjxtT7s1wOggcAAD/Vrl07zZ8/X7m5uSopKdH8+fO1cOFCZWVl1fu1oqKi6oyNHz9e77zzjlwulzZt2qRjx45p7NixF32NS80vLi5WZGSkbr75Zt1yyy265ZZb1Lt3bz333HPu2CoqKqpzRelrrrnG4xyfxsIhLQAAmoCIiAj9/ve/14cffqivv/5akhQaGipJqqqqcs8rLy+X0+n06jXHjh2rmTNn6uuvv9by5cv1L//yL4qJibms+fHx8Tp58qQmTJhw0cNuiYmJdb5+XlBQoNraWq/W2xDs4QEAwA/t2rVLW7du9RgrLS1Vbm6uOnfuLOnsHqDw8HB988037jlLlizx+j1iY2M1dOhQvfnmm1q9evVFD2d5M//ee+9VdXV1nRObt2/frv3790uSRowYoePHj2v16tXux1955RWv19sQ7OEBAMAPBQcHa/r06Tp58qRSU1Plcrm0ceNGderUSc8884yks9fKefrppzVt2jTl5OToxIkTklSvQ0Tp6el68MEHddVVV2nkyJGXPT8xMVHvvvuuJk6cqJycHHXp0kX79u1TQECAVq1aJUlq37695s2bp/T0dL399tsqLS2V0+lUUFDj5wjBAwCAH0pNTdXXX3+tvXv36vvvv5ckPfnk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jh07pq1bt7rH1qxZo5YtW6pfv34WrgwAAPiLJr+Hp0+fPho1apTS09M1b948lZSU6KmnntLcuXMVHh5u9fIAAIAfaPLBI0krVqzQq6++qrfeekuhoaFatGiR0tPTrV4W/FRoaKjmzp1b57AmgKaP7RsXE+Dy5rtcAAAATViTP4cHAADg1xA8AADAeAQPAAAwHsED/MKiRYv0hz/8weplAMZbv369/vVf/1X9+vVTTk6O1cth228GjPiWFuArBQUF2rNnj9XLAIxWWFioUaNGaf78+br55pvVqVMnq5fEtt8MEDwAgCsqNzdXtbW1euSRR6xeCpoRggdNzquvvqqCggLdcMMN+vjjj1VcXKx77rlHM2bM0JIlS/T+++9LkiZMmKAxY8a4n7dq1SrNnz9fktSqVSt1795ds2bNUps2bS75fv/4xz/0yiuvaO/evWrXrp0eeOABDR8+vPE+IGCwN998U/Pnz1dNTY369u0rSdq8ebNCQkJ+dVtj20dDEDxocvLz87Vw4UINGzZM06ZNU35+vjIyMpSdna3k5GRlZGQoLy9P999/vxITE9W/f39J0oABA5SQkCBJstvtWrx4sfr166c9e/Zc9CJl27dv1x133KGMjAylpaVp//79mjBhgubPn6+JEydesc8MmGLYsGGy2+2aPXu2Xn31VUlnfwTam22NbR8N4gKamH/7t39zxcXFuRwOh3vst7/9rSspKclVXV3tHhsyZIjrj3/840Vfx+l0uuLj413vv/++e+zJJ590DRo0yH3/tttuc2VmZno874033nB16tTJB58EaJ7Wrl3rCg0N9RjzZltj20dDsIcHTVK3bt0UEhLivh8XF6eAgAAFBQV5jBUWFrrvV1VV6a233tKnn36qwsJCOZ1O2e127d+//4Lv4XQ6tXnzZh05ckSbN2+Wy+WSy+VyP6eqqspjDQAuT322NbZ9XC6CB01ScHCwx/2AgIALjrl+8cspGRkZ2rhxozIzM5WcnKywsDA9+OCDOnPmzAXfw+FwqKamRhMnTtTgwYPrPP7L/8ACuHz12dbY9nG5+H8Nzcbq1au1cOFC3XfffZKk6upqFRUVXXR+eHi44uLiZLfb3SdXAvC9xt7W2PYhceFBNCNXX321du7cKUlyuVyaM2eOTp48ecnnPPLII/rzn/+sr776yj32888/67XXXmvUtQLNTWNua2z7kNjDg2bk5ZdfVnp6ulatWqVTp04pKSlJnTt3vuRz/vjHP+rUqVO68847FRsbq9raWtlsNr300ktXaNVA89CY2xrbPiQpwPXLA51AE/Dzzz+rsrJS1113nXvswIEDqqmpUUpKints3759kuRxFdfTp0/rp59+UmRkpJKTk7V7925FRUUpPj5eknTo0CGVlZXp+uuv93jPM2fOKC8vT1FRUUpKSmrMjwcYr7S0VHl5eerdu3edxy61rbHtoyEIHgAAYDzO4QEAAMYjeAAAgPEIHgAAYDyCBwAAGI/gAQAAxiN4AACA8QgeAABgPK60DKDJ+uGHH7Rv3z5FR0erR48eCg8Pt3pJAPwUwQOgybHb7Ro1apS+/fZb9e3bV+Xl5fr55581c+ZMzZgxw+rlAfBDBA+AJmfu3Lk6ePCg9u/fr8jISElnf65g3bp1debm5+fr22+/VevWrdWzZ0+1aNFC0tloWr9+vQYMGOD+eQFJWrdundq2batbbrnlynwYAFcEPy0BoMm5/fbb1bp1a61cufKic1wul2bMmKEVK1aob9++On78uI4dO6YPPvhAPXv2lCSNHDlSxcXF2rx5s2w2m9555x1NnjxZ27ZtU2pq6pX6OACuAE5aBtDk9O3bV2vXrtWiRYt09OjRC855/fXX9cknn+jHH39UTk6Otm7dqoceekgTJkxwz1m8eLH279+v5557Tvn5+frDH/6gl156idgBDMQeHgBNTmVlpebMmaMlS5aouLhYiYmJGjZsmP793/9d7du3lyT17t1bSUlJGjNmjFwul1wulw4fPqzHHntMRUVFiomJkSR98sknuvvuu9W5c2clJycrJyfHyo8GoJEQPACarNraWu3Zs0dffvmlXn75ZZWXl2v37t2Kjo5WmzZtlJSUpE6dOtV53vz585WQkOC+P2jQIG3evFl5eXlKSUm5kh8BwBVC8AAwwt69e9WlSxe9++67SktL07XXXqu0tDS98MILl3zemjVrlJaWpvbt26tPnz7Kzs6+QisGcCVxDg+AJufAgQN1xiorKyVJUVFRkqRhw4YpOztbp0+f9ph3+PBh9/8+evSoHn74Yc2bN0/vv/++Vq9erRUrVjTewgFYhj08AJqcKVOm6Pvvv9edd96p5ORkHTlyRK+//rqSkpK0YcMGhYSE6MSJExo4cKBcLpcefvhhhYWFacuWLSooKNDGjRvlcrk0bNgwVVdX67PPPlNgYKD+4z/+Q08//bS+/fZb97lAAMxA8ABokrZt26acnBzl5+fr6quvVp8+ffS73/1OwcHB7jkVFRXKzs7W3//+d4WHh6tv374aO3asbDabtm3bpgULFuhPf/qTEhMTJZ39KvsTTzyhDh06aPr06VZ9NACNgOABAADG4xweAABgPIIHAAAYj+ABAADGI3gAAIDxCB4AAGA8ggcAABiP4AEAAMYjeAAAgPEIHgAAYDyCBwAAGI/gAQAAxiN4AACA8f4P05rWGgJt69gAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data = df , x = 'Sex', hue = 'Survived')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2d8a1292-fd56-4f68-b907-6b67ac5f282e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA1MAAAK8CAYAAAAK3eV/AAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlcelbwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAT21JREFUeJzt3XmcX+PdP/7XyCZBEoJImCwizWKLRJDETspN0ya0dopqFW0tlaILUm6ppUpt1dtS3SxJSG2lRRGUEkRUSVCSEFmETGSPfH5/+Jpfp1lMjkk+k+T5fDw+jzuf67rOOe9zZh5Tr/s65zoVpVKpFAAAAFbIOuUuAAAAYHUkTAEAABQgTAEAABQgTAEAABQgTAEAABQgTAEAABQgTAEAABQgTAEAABQgTAGw0s2cOTNnnnlm+vbtm65du+YPf/hDuUuqtSeeeCJdu3bN888/X+5S6iXXB1ibNSx3AQDUDw8//HBOOeWU3H777dl+++3rdN8nn3xynnrqqdxyyy3ZdNNNs9lmm9Xp/lemjz76KK+99lrmzJlT7lLqJdcHWJsJUwAkSWbNmpXXXnstc+fOrfN933vvvfnGN76R3Xffvc73vbLttttu+de//pX27duXuxQA6hlhCoCVas6cOamqqkrz5s3LXUoh6623Xrp27VruMgCohzwzBbCKLFy4MNdff30OPPDAbL/99hkwYED+8Ic/pFQqVY+ZOXNmzj///Oy2227Zfvvt85WvfCV33XVXjf3cfPPN6dq1az788MMa7a+++mq6du2a+++/v7rt/vvvT9euXfPqq69m5MiR2XfffbPDDjvkpJNOyrRp06rH/fa3v813vvOdJMlhhx2Wrl27pmvXrksc+799Vr1XX311dthhh+p/f7rfjz/+eKn7W7x4cW688cYceOCB2WGHHXLAAQfk5z//eebNm/e5z3/48OH54he/mO7du+eRRx5J165dM2zYsCVqmDx5crp165Zrr702yZLPBL3wwgu13jb55Od+3XXXZf/99892222XffbZJ9dee+0yr8HSar/11luz1157pWfPnvnOd76TKVOmLDG+NsdZ1vWYPn36Muuoze/tf/v0Z9G1a9d069YtO+ywQw455JA88MADNcbV5uddmzEA5WJmCmAVmDdvXvbbb7+MGTMmQ4YMSZ8+fTJt2rTcfvvtWbBgQY477rh8+OGH6dOnTz766KNceuml6dChQ+6+++4cfPDB+dGPfpQLLrggSfL+++/ntddey6JFi5Y4xmuvvZaqqqrqtqqqqrz22mu56aabMnv27AwZMiRTp07NKaeckpdffjmjRo1KkgwYMCCzZs3Kd77znVx88cXVz0y1adNmmedUm3oPO+yw9O3bN7169cphhx1WHdgaNGiw1H2ee+65+eUvf5nLL788vXr1yvTp0/PQQw/l+9//fq655prC53/ddddl3rx5Offcc/P444+nW7duWbhwYa655pp87Wtfq7Gf3/zmN3nttdey3377JVnymaAePXrUetv58+dn//33zz//+c9ccMEF6dWrV1555ZUMHjw4TzzxRP74xz8u8/p+Wvs111yTRYsW5X//938zderUDB48OPfff3+ee+65bLTRRit0nGVdj/++lv95TT/r93ZpOnbsmJEjR1Z/f//993PnnXfmgAMOyIgRIzJo0KAktft512YMQNmUAFjpfvrTn5aSlB5//PEl+hYsWFAqlUqlwYMHlyoqKkovvvhijf5TTz21VFFRUXrllVdKpVKpdOmll5aSlKZNm1Zj3AsvvFBKUrr11lur22699dZSktJhhx1WY+zNN99cSlJ69tlnq9vuuuuuUpLS3//+91qdU23rnTVrVilJ6bzzzvvMfXbp0qV09NFHL9H+6TUqlYqd/8CBA2uMXbx4cenCCy8sVVRUlF5//fUafZ07dy7tueee1d///Oc/l5KURo0aVd1W220vuuiiUkVFRempp56qMe7BBx8sJSn99a9/Xea1+LT2QYMG1Wh//fXXSw0aNCidccYZK3yc5V2PpanN7+3Srs+yfPWrXy3tuOOO1d9r8/OuzRiAcnGbH8AqMGzYsGy77bbZbbfdluhr1KhRkuS+++5Ljx49llhJ77jjjkupVMp9991X+PiHHnpoje877bRTkuS1114rvM+VUe8WW2yRBx98MHfffXfmz59f3f7pNSrq8MMPr/G9oqIixx13XNZZZ53cdNNN1e2PP/54xo8fn2984xvL3V9tt73jjjvSvXv39OnTp8b2X/ziF7Peeuvlz3/+82fWfsQRR9T43qlTp/Tr1y/33ntv4eMs7XosTW1+b5fl8ccfz9e//vX07t073bp1S9euXfPII4/k1VdfrR5Tm5/3yvqdAKgLwhTAKjBp0qR07NhxuWPefffdtGvXbon2T1eRe/fddwsfv7Kyssb3T28PW96zMp9lZdR73XXXpVu3bhk4cGA23HDD7L333vnFL37xuVcYXFqdbdu2zf77759bbrml+rmim266KS1btszBBx+83P3Vdtu33347b7/9drbZZptss8022XrrrdO9e/d069YtCxYsyHvvvfeZtf/3z+7TtsmTJxc+ztKux9LU5vd2aYYNG5Y999wzjRs3ztChQzNs2LCMHDkyBx54YI2fZW1+3ivrdwKgLnhmCmAVaN68+WcGlw022CAffPDBEu3vv/9+dX/yyepySZb4j8mpU6cuc9/LekaptJxFBD5LbetdEZ07d86jjz6a9957L08++WQeeOCBnH322Rk+fHiefPLJJMXOf911111q+wknnJBBgwblwQcfzG677ZZhw4bl2GOPTdOmTT+z1tpsu8EGG2SLLbbIjTfeuNR91GaFwxkzZiy17T+v74oeZ1nXY2nbFQncv/jFL9KjR4/83//9X4322bNn1/hem593bcYAlIswBbAK7L333rntttvy7rvvpm3btksd07dv39x333354IMPsuGGG1a3P/TQQ0mSfv36Jfn/Z35ef/31GrMWjz766Oeq8dP/wF7WYgRF6y1is802y8EHH5yDDz44rVq1ysUXX5wpU6akdevWdXr+X/rSl9K6devceOONmTRpUubMmfOZt/ityLZ777137rrrrrRu3brGNVoRjz76aP7nf/6n+vucOXPyj3/8I3vttVedHmdpavN7uzSzZs1aYvarqqoqjzzyyFLHL+/nvSJjAFY1t/kBrAI//vGPs+666+awww7L22+/nST5+OOPc+edd1Y/W3TWWWdlwYIFOeGEE/LRRx8lSZ577rmcd9556du3b/r3758k2WuvvdK6desMHTq0ennoBx54IGPGjPlcNXbq1ClJ8uKLL9ZqfG3rXRHf+ta38sILL1TPmM2aNSvPPfdcWrduXX1rYl2ef8OGDfP1r38999xzT375y1+mR48e6dmzZ51te95556VRo0Y55JBD8tZbb1W3T5s2LT/72c+qV1Ncnueffz5//etfk3yyat8pp5ySmTNn5qyzzqrT4yxNbX5vl2aPPfbIo48+mtGjRyf5JEgde+yx6datW41xtfl512YMQLkIUwCrwJZbbpknn3wy6667bjp16pS2bdtmww03zG233Va9gMMOO+yQBx54IK+//no22mijtGnTJv369cu+++6be++9t3qRgKZNm+a3v/1tXn755bRq1SqtW7fObbfdlnPPPfdz1di5c+d873vfy+mnn54OHTp85numalvvith9993zrW99KxtssEE6duyYTTfdNKVSKX/+85+rFxyo6/P/xje+kYULF+af//xnrWelartthw4d8vTTT2e99dZLly5dstlmm2WTTTbJdtttl1mzZmXrrbf+zGOcf/75uf7669O6deu0aNEiDz30UEaMGJEdd9yxTo+zNLX5vV2a//3f/03//v2z0047pbKyMl26dMnAgQPTt2/fGuNq8/OuzRiAcqkofZ4b5gFYYR999FGmTZuWzTffPI0bN17qmOnTp6eqqipt2rRZ5vM7ixcvzoQJE7LhhhumRYsWmT9/fv79739n8803r36eZtasWXnnnXfSsWPHNGnSpHrbjz/+OOPHj1/qbWGzZ8/O5MmTs2jRorRp0yYtWrT4zHNaXr2lUimvvfZaNt5442y88cafua/kk+ehpk6dmtatWy/z+Z7Pc/7/7fXXX8+iRYvSoUOHJY43e/bsTJw4Me3bt1/qz2J52/6n+fPn5913303Lli1rdSvebbfdlsMPPzwvvPBCevTokQ8++CBVVVWprKzMOuss+/8Xurzj1PZ6LM2yfm+Xd30++uijzJgxI23btk3Dhg0zbdq0zJgxI126dKkxrjY/79qMAVjVhCkAqIf+O0wBUP+4zQ8AAKAAYQoAAKAAt/kBQD30eZ5vAmDVEKYAAAAKcJsfAABAAQ3LXUB9sXjx4rz77rvZYIMNCr0bBQAAWDOUSqXMmjUrbdu2Xe7rKISp/+fdd99NZWVlucsAAADqiYkTJ2aLLbZYZr8w9f98+oLHiRMnpnnz5mWuBgAAKJdPX5L+aUZYFmHq//n01r7mzZsLUwAAwGc+/mMBCgAAgAKEKQAAgAKEKQAAgAI8MwUAAGuoxYsXZ8GCBeUuo95p1KhRGjRo8Ln3I0wBAMAaaMGCBfn3v/+dxYsXl7uUeqlly5bZbLPNPtc7ZoUpAABYw5RKpUyePDkNGjRIZWXlcl88u7YplUqZM2dOpk6dmiRp06ZN4X0JUwAAsIZZtGhR5syZk7Zt26ZZs2blLqfeadq0aZJk6tSp2XTTTQvf8ieiAgDAGubjjz9OkjRu3LjMldRfn4bMhQsXFt6HMAUAAGuoz/M80JquLq6NMAUAAFCAMAUAAKwy3/nOdzJw4MCVfpyzzz47+++//0o9hjAFAABrkfnz5+f8889P586d06xZs3Tp0iU/+MEPMmXKlHKXttoRpgAAYC1y+umn5+abb85NN92U999/P/fff3/atGmTq666apUc/+qrr87IkSNXybFWNmEKAADWInfddVdOPPHE7LbbbmnatGk6deqU008/PRdeeGH1mC996Us57bTTamx35pln1rht7thjj82gQYNy4oknZqONNkrnzp1z3333Zd11182HH35YY9sf/OAH6dWrV5Kat/nVZnyS3HLLLdl6662z7rrrpnPnzrnkkkuqVyxMksWLF+ess85Kq1atsummm+bQQw/N9OnTP8dVqh1hCgAA1iIbbbRRnnjiicyZM+dz72vkyJHZfPPN8+abb2b8+PHZb7/90rx58wwfPrx6TKlUyq233pqjjjpqie1rM/6mm27Kj370o1x99dWZMWNGbrvttvz617/OL37xi+ptfvGLX+Q3v/lN7rzzzowbNy49evTIjTfe+LnP77MIUwAAsBa54oor8swzz2SzzTbL//zP/+SnP/1pXnzxxUL72nrrrXPuueemZcuWSZKGDRvmkEMOyR/+8IfqMY899lgmT56cww8/fIntazN+yJAhufDCC7PXXnulWbNm6dWrV374wx/WCEuXX355zjrrrOyxxx5p2bJlzjnnnBozWyuLMAUAAGuR/v3756233srNN9+cbt26ZcSIEenZs2fOO++8Fd5X9+7dl2g76qij8thjj2XSpElJkj/84Q/ZZ599stlmmy11H8sbP23atEyYMCEnnHBCGjZsmAYNGmSdddbJN77xjbz55ptJkqqqqrz77rvZcccda+y3d+/eK3w+K0qYAgCAtcwGG2yQgw8+OJdffnnGjBmTs88+OxdccEGmTp26zG0WL168RFvjxo2XaNtll13SqVOn3HrrrVmwYEGGDx++1Fv8ajP+02Pef//9WbRoUT7++OMsXrw4pVIp8+fPT/LJbYFJeV5QLEwBAMBabvfdd0+pVKpeCGLDDTfMBx98UGPM+PHja72/I444In/4wx9y3333ZcGCBRk0aFCh8a1bt87mm2+eP//5z8vctkWLFmnbtm2effbZGu3//X1lEKYAAGAt8sUvfjG///3v89Zbb2XevHkZM2ZMfvrTn6Z79+7p3LlzkmS33XbL3Xffneeeey6zZs3KjTfemPvuu6/WxzjyyCMzZsyYnH/++Rk4cGDWX3/9wuPPP//8XHvttbnuuusyY8aMvPvuu/nd736XM844o3rM6aefnosvvjiPP/54Zs6cmaFDh2b06NEreGVWnDAFAABrkUsuuSQPP/xw9tprr2y44Yb58pe/nO7du+fBBx+svlXu+OOPz5FHHpn/+Z//yVZbbZXHHnssxx57bK2P8YUvfCG9e/fOSy+9tNxb/Goz/oQTTshvfvOb/N///V/atm2bnXbaKY888ki+973vVY85/fTTc/TRR+crX/lKOnfunBdeeCHHH398restqqL06U2Ga7mqqqq0aNEiM2fOTPPmzctdDgAAFDZv3rz8+9//TseOHbPuuuuWu5x6aXnXqLbZwMwUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAQ3LXQAAwNqo1+DflrsElmH0pceUuwRWE2amAAAAChCmAAAAChCmAAAAChCmAAAACrAABQAAsFqYMGFCrrjiirzxxhvp2LFjTj311HTs2LFs9QhTAADAKl9hckVXTXznnXfSu3fv9OvXL4cddljuvPPO9O7dO6NHj0779u1XUpXL5zY/AACg3rv44ouz4YYbZtiwYTn88MNz++23Z7PNNstFF11UtpqEKQAAoN574IEHMmDAgDRo0CBJss4662TgwIF54IEHylaTMAUAANR7b731ViorK2u0bbHFFpk4cWIWL15clpqEKQAAoF4rlUpZuHBhmjZtWqO9WbNm1X3lIEwBAAD1WkVFRVq0aJEZM2bUaH///ffTtGnTNGnSpCx1CVMAAEC916NHj7zwwgs12l544YVsv/32ZapImAIAAFYDX//613Pvvffmn//8Z5Lk1VdfzV133ZWvf/3rZavJe6YAAIB679hjj83TTz+dnXbaKdtss01efvnlHHLIIfnWt75VtpqEKQAAYIVforuqVVRU5Prrr88Pf/jDvPHGG+nYsWM6duxY1pqEKQAAYLXRvn37tG/fvtxlJPHMFAAAQCHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAsNp45plncuWVV2bMmDHlLiUNy10AAABQfhN+uu0qPV67c8eu0Ph//OMf+fa3v50GDRrkpZdeys9//vNsv/32K6m62jEzBQAA1HuNGjXKDTfckGeffTZNmzYtdzlJ6snM1OTJk/PRRx+lXbt2adKkSY2+CRMmZOrUqTXamjZtmq233nqJ/UydOjVTp07NlltumWbNmq3UmgEAgFVnhx12KHcJSyhrmPrzn/+cs88+Ox9++GEaNmyYadOm5eyzz84Pf/jD6jEXXXRR7rjjjmy55ZbVbVtttVVuu+226u8LFizIcccdlxEjRqRNmzaZNm1afvGLX+Sb3/zmKj0fAABg7VHWMPXGG29kxIgR2WqrrZJ8Eq4OOOCA9OnTJ3vttVf1uL333jvDhw9f5n4uvPDC/O1vf8v48eNTWVmZ22+/PYcffnh69uyZXr16rfTzAAAA1j5lfWbqO9/5TnWQSpL+/funYcOGefvtt2uMW7hwYV555ZW88847S93PDTfckBNOOCGVlZVJkkMPPTTdunXLjTfeuPKKBwAA1mplf2aqqqoq48aNS1VVVW644YZ069YtgwYNqjHmnnvuyb/+9a9MnTo1rVq1yvXXX5999903ySfPW02ePDm9e/eusc3OO++c559/fpWdBwAAsHYpe5j65z//me9+97uZPn16Zs6cmauuuiotWrSo7t9nn33yk5/8JJtvvnkWLlyYwYMHZ9CgQRk7dmw6dOiQ999/P0nSqlWrGvvdeOONq/uWZv78+Zk/f37196qqqjo+MwAAYE1W9jDVp0+fPPfcc0mSBx98MAMGDEjTpk1z8MEHJ0m+9rWvVY9t1KhRLrvsstx8880ZOXJkTjvttDRq1ChJagSjJJk7d25139IMHTo0Q4YMqevTAQAAVoKpU6fmj3/8Y5JP/tv/0UcfzaJFi9K5c+cceOCBZampXr1nar/99svOO++ce+65Z5ljGjZsmE022SSTJk1KkmyxxRapqKjIu+++W2Pcu+++m3bt2i1zP+ecc05mzpxZ/Zk4cWLdnAQAAFDn5s+fn7feeitvvfVWTjzxxGyxxRZ56623Mm3atLLVVLaZqcWLF2fRokVp3LhxdduiRYsyadKk6uefSqVSFixYUOPdU2+++WbeeuutdOvWLUmy3nrrZZdddsl9992XI488Mskns1IPP/xwfvSjHy3z+E2aNFninVYAALC2anfu2HKXsFyVlZW54ooryl1GDWULU/Pnz88uu+ySk046Kd27d8+HH36Y66+/PjNnzszJJ5+c5JNV/HbcccecdNJJ2XrrrTNhwoRccMEF2WabbXLEEUdU7+uCCy7I/vvvny5duqRPnz658sor07Jly5x44onlOj0AAGANV7bb/Jo2bZq7774748aNy3nnnZdf//rX6dmzZ/71r39VL5feuHHj3HfffXn99ddz/vnn56677spJJ52UZ555Jk2bNq3e1z777JMHH3wwY8aMyZAhQ1JZWZknnngizZs3L9fpAQAAa7iyLkDRvn37XH755csd065du88ck3zyYt+99967rkoDAABYrnq1AAUAAMDqQpgCAAAoQJgCAIA1VKlUKncJ9dbixYs/9z7K/tJeAACgbjVq1CgVFRWZNm1aNtlkk1RUVJS7pHrj09cvTZs2Leuss06NVzWtKGEKAADWMA0aNMgWW2yRSZMm5a233ip3OfVSs2bN0q5du6yzTvGb9YQpAABYA62//vrp3LlzFi5cWO5S6p0GDRqkYcOGn3vGTpgCAIA1VIMGDdKgQYNyl7HGsgAFAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAcIUAABAAQ3LXcCcOXMyevTofPTRR+nWrVs6dOiwxJjFixfnmWeeyZQpU7LNNttkq622KjQGAACgrpR1ZurGG2/M1ltvnfPPPz9XXHFFtt566xx//PFZvHhx9ZiZM2emX79++epXv5prrrkm22+/fc4+++wa+6nNGAAAgLpU1pmpDTfcMC+99FI22GCDJMnYsWOz3XbbZdCgQRkwYECS5Ec/+lFmzJiRV155JS1atMgTTzyR3XbbLf37988+++xT6zEAAAB1qawzUwcddFB1kEqSLbfcMg0bNsysWbOSJKVSKX/4wx/yjW98Iy1atEiS7Lrrrtlpp53y+9//vtZjAAAA6lrZn5l677338uijj6aqqiq///3vc8ABB+SrX/1qkmTSpEn58MMPs80229TYZtttt82LL75Y6zFLM3/+/MyfP7/6e1VVVd2cEAAAsFYo+2p+U6ZMyciRI3PHHXfktddeS+/evdOw4ScZb+bMmUmSjTbaqMY2rVq1yocffljrMUszdOjQtGjRovpTWVlZR2cEAACsDco+M7X99tvntttuS5K88sor2WmnndK6det885vfTJMmTZIkH330UY1tPvroo6y77rpJUqsxS3POOefkjDPOqP5eVVUlUAEAALVW9pmp/9S9e/f06tUro0aNSpK0a9cuDRs2zIQJE2qMe/vtt7PlllvWeszSNGnSJM2bN6/xAQAAqK2yhamPP/4406dPr9H20UcfZdy4cdUzRE2aNMk+++yTYcOGVY+ZPn16HnnkkRx44IG1HgMAAFDXKkqlUqkcB543b1569uyZAw44IN27d8+HH36YW265JXPmzMkTTzyR1q1bJ0lefPHF7Lrrrhk0aFD69OmTG264IRUVFXnqqaeqb/GrzZjPUlVVlRYtWmTmzJlmqQCAla7X4N+WuwSWYfSlx5S7BMqsttmgbDNT6667bp599tlsueWW+fvf/54JEybkjDPOyD//+c/qIJUkPXr0yPPPP582bdrkmWeeyRFHHJHHH3+8RkiqzRgAAIC6VLaZqfrGzBQAsCqZmaq/zExR72emAAAAVmfCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAHCFAAAQAENy13A6NGj89RTT6Vhw4bp169ftttuuxr99957b5577rkabZtuumlOPvnkGm0zZ87MnXfemSlTpmTbbbfNAQcckIqKipVePwAAsHYq28zU4sWLs9dee+Xb3/52xo8fn9GjR6dPnz754Q9/WGPcvffemzvuuGO5+5owYUK23Xbb/PrXv84777yTb33rWznooINSKpVW5ikAAABrsbLNTFVUVOS8887LnnvuWd325S9/OV/5yldy9NFHp1u3btXt3bt3z/nnn7/MfZ111llp06ZNRo0alYYNG+a73/1uunfvnuHDh+drX/vaSjwLAABgbVW2mamKiooaQSpJdt555yTJW2+9VaN9woQJueSSS3L99dfn5ZdfrtH38ccf509/+lOOPvroNGz4STb8whe+kN133z3Dhw9fafUDAABrt3q1AMVtt92Wxo0bp2fPnjXaP/7447z33nt55JFH0qtXr/z4xz+u7pswYULmzp2bzp0719imc+fOee2115Z5rPnz56eqqqrGBwAAoLbKvgDFp5555pmcffbZGTJkSFq3bl3dftppp6Vr167V3++55558+ctfzn777ZfddtstH330UZKkRYsWNfbXsmXL6r6lGTp0aIYMGVLHZwEAAKwt6sXM1IsvvpgDDjgg3/zmN3P22WfX6PvPIJUkAwYMSNu2bfPoo48mSdZff/0kn6zm958+/PDD6r6lOeecczJz5szqz8SJE+vgTAAAgLVF2cPUSy+9lH333TeHH354fvnLX9Zqm1KplDlz5iRJKisr07Rp04wfP77GmPHjx6dLly7L3EeTJk3SvHnzGh8AAIDaKmuYGjt2bPbZZ58cdthhufrqq5fo//jjj/PCCy/UaLvnnnsyefLk7LXXXkmShg0b5itf+Up+97vfZdGiRUmScePG5fHHH8/BBx+88k8CAABYK5Xtmak5c+Zkn332yTrrrJONN964xtLnX/7yl9OzZ8+USqWccsopadmyZbbeeutMmDAhf/rTn/L9738/X/ziF6vHX3zxxenXr19233337LjjjrnzzjszYMAAy6IDAAArTVkXoDj55JOX29+wYcM89dRTeeSRR/LCCy+kW7duGTJkyBLPUbVr1y4vv/xyRowYkSlTpuRXv/pVDjzwwFRUVKzM8gEAgLVYRalUKpW7iPqgqqoqLVq0yMyZMz0/BQCsdL0G/7bcJbAMoy89ptwlUGa1zQZlX4ACAABgdSRMAQAAFCBMAQAAFCBMAQAAFCBMAQAAFCBMAQAAFCBMAQAAFCBMAQAAFCBMAQAAFCBMAQAAFFAoTD366KN1XAYAAMDqpVCY2nfffVMqleq6FgAAgNVGoTDVoUOHjBs3rq5rAQAAWG0UClM//vGPc+yxx+bpp59OVVVV5s2bV+MDAACwpmtYZKPjjjsuSdKnT5+l9rsFEAAAWNMVClOjRo2q6zoAAABWK4XC1K677lrXdQAAAKxWCr9nauHChXnyySfzu9/9rrrt/fffr5OiAAAA6rtCYWrChAnp2bNn9tlnnxxzzDHV7d/85jdzzz331FlxAAAA9VWhMHX66aenT58+qaqqqtF+5pln5mc/+1mdFAYAAFCfFXpm6rHHHsurr76axo0b12jfdtttM3r06DopDAAAoD4rNDM1b968rLPOJ5tWVFRUt0+ePDnNmjWrm8oAAADqsUJhas8998yvfvWrJP9/mJo9e3YGDx6cfffdt+6qAwAAqKcK3eZ32WWXZffdd8/999+fUqmUr33ta9XvnnryySfrtEAAAID6qNDMVNeuXfPyyy9nv/32y4ABAzJv3ryceOKJGTNmTDp16lTXNQIAANQ7hWamkmTTTTfNT37yk7qsBQAAYLVRKExNmjRpmX1NmjRJq1atqheoAAAAWBMVClOVlZXL7W/evHm+/vWv57LLLlti+XQAAIA1QaHpo6uvvjqVlZX59a9/neeeey6jR4/O9ddfn8033zw///nPc+211+buu+/OBRdcUNf1AgAA1AuFZqZuvPHGDBs2LDvvvHN1W8+ePbPddtvllFNOyejRo1NZWZlvfOMbAhUAALBGKjQz9eqrr6ZLly5LtHft2jWvvvpqkk/C1Xvvvff5qgMAAKinCoWpdu3a5aqrrlqi/Yorrki7du2SJP/85z/To0ePz1UcAABAfVXoNr8rrrgigwYNyu23355evXqlVCpl9OjRefPNN3PXXXclSYYPH54LL7ywTosFAACoLwrNTO2///4ZP358Bg0alNmzZ2fu3Lk56KCDMn78+Oy///5JkksvvTR77LFHnRYLAABQXxR+ae8WW2xhcQkAAGCtVThMLV68OJMmTcqMGTOW6POsFAAAsKYrFKaeeuqpHHHEEXn77beX2l8qlT5XUQAAAPVdoTB1yimnZL/99suZZ56ZDTfcsK5rAgAAqPcKhalx48blscceS/Pmzeu6HgAAgNVCodX8ttpqq0ydOrWuawEAAFhtFApTZ555Zk444YSMHTs2c+fOzbx582p8AAAA1nSFbvM75phjkiTbbbfdUvstQAEAAKzpCoWpUaNG1XUdAAAAq5VCYWrXXXet6zoAAABWK4WemUqShQsX5sknn8zvfve76rb333+/TooCAACo7wqFqQkTJqRnz57ZZ599qp+fSpJvfvObueeee+qsOAAAgPqqUJg6/fTT06dPn1RVVdVoP/PMM/Ozn/2sTgoDAACozwo9M/XYY4/l1VdfTePGjWu0b7vtthk9enSdFAYAAFCfFZqZmjdvXtZZ55NNKyoqqtsnT56cZs2a1U1lAAAA9VihMLXnnnvmV7/6VZL/P0zNnj07gwcPzr777lt31QEAANRThW7zu+yyy7L77rvn/vvvT6lUyte+9rXqd089+eSTdVogAABAfVRoZqpr1655+eWXs99++2XAgAGZN29eTjzxxIwZMyadOnWq6xoBAADqnUIzU0my6aab5ic/+Un191mzZmX99devk6IAAADqu0IzU//6179y1llnVX8/9dRT07x587Rt2zYvvvhiXdUGAABQbxUKU2eccUb69++fJBk3blxuvPHG3HvvvTn88MPzgx/8oE4LBAAAqI8K3eb31FNPZfjw4UmSBx98MF/5yldy4IEHpl+/funYsWOdFggAAFAfFZqZWnfddfPee+8lSf785z9n7733TpIsWLBgiRf5AgAArIkKzUwdeOCBOeigg9KrV6+MGjUqv/nNb5IkDz/8cPXtfwAAAGuyQjNTV199db785S9n0aJFueeee7LpppsmSUaNGpXzzjuvTgsEAACojwrNTDVr1iwXXHBBjbZZs2blmmuuSUVFRZ0UBgAAUJ9ZGh0AAKAAS6MDAAAUYGl0AACAAiyNDgAAUICl0QEAAAqwNDoAAEABdbY0epJce+21n7sgAACA1UGhMFVXFi1alBEjRuSpp55Kw4YNs+uuu2bgwIFLvKvqjTfeyE033ZQpU6Zk2223zbe+9a00bdp0hccAAADUlUK3+SXJrbfemoMOOig77rhjevToUeNTG4sXL0737t0zcuTIdOrUKZtssklOOumkHHbYYTXGvfTSS9lhhx3y5ptvZtttt81NN92UPfbYIwsWLFihMQAAAHWpUJi6/PLLc9ppp6V79+4ZPXp0vvrVr6Zdu3YZM2ZM+vbtW6t9VFRU5C9/+UtuvfXWfO9738vZZ5+dYcOG5Y477sjzzz9fPe7ss89Onz59cuutt+bUU0/NX//614wdOza33HLLCo0BAACoS4XC1K9+9asMGzYsF154YZLkxz/+ce6+++5cfvnlmTBhQq32UVFRkQ4dOtRo23LLLZMk06dPT/LJUusPPfRQDjnkkOoxm266afbee+/ce++9tR4DAABQ1wqFqX//+9/ZZZddknzyzqlZs2YlSY477rg89thjhYu59tprs8EGG2SnnXZKkkyYMCELFy5M+/bta4xr37593nzzzVqPWZr58+enqqqqxgcAAKC2CoWpRYsWVb+ct127dhk9enSS5J133knDhsXWtBg5cmQuvvjiXHfddWnZsmWSZO7cuUmS9ddfv8bYDTbYoLqvNmOWZujQoWnRokX1p7KyslDdAADA2qnwAhSf+vrXv55DDz00Rx11VPbbb78MGjRohffxwAMP5LDDDsvPf/7zHHnkkdXtLVq0SJJ88MEHNcbPmDGjuq82Y5bmnHPOycyZM6s/EydOXOG6AQCAtVehMDV58uTqf59zzjm56KKLsu666+a0007Lddddt0L7evDBBzNo0KAMHTo0p556ao2+ysrKtGzZMi+//HKN9rFjx2bbbbet9ZiladKkSZo3b17jAwAAUFuFwtRmm21W/e+Kiop84xvfyA033JAzzzwzTZo0qfV+/vrXv2bgwIG56KKLcvrppy/RX1FRkcMOOyw33nhj9XNZTz31VP7xj3/kiCOOqPUYAACAulZRKpVKRTacPXt2/vjHP+Zf//pXkqR79+454ogj0qxZs1ptP2vWrLRu3TobbLBB+vfvX6PvhBNOyJ577pnkk9v3vvjFL2bq1KnZeuutM2rUqHzrW9/Kz3/+8+rxtRnzWaqqqtKiRYvMnDnTLBUAsNL1GvzbcpfAMoy+9Jhyl0CZ1TYbFApTzz33XA488MB8/PHH1bfSjR07No0aNcp9992Xnj17fuY+5s+fn2HDhi21r0+fPunUqVP1948//jhPPvlkpkyZkm233TZdu3ZdYpvajFkeYQoAWJWEqfpLmGKlhqkdd9wxPXv2zC9/+cusu+66SZJ58+ble9/7Xl544YU8++yzxSsvE2EKAFiVhKn6S5iittmg0Drm//znP/PAAw9UB6nkk/dNXXTRRWnXrl2RXQIAAKxWCi1A0alTp7z33ntLtL/33nvZcsstP3dRAAAA9V2hMHXGGWfk0EMPzf33359p06Zl6tSpuf/++3PIIYfk+9//fubNm1f9AQAAWBMVemaqoqKi1mMLLha4ynlmCgBYlTwzVX95ZoqV+szUqFGjChcGAACwJigUpnbddde6rgMAAGC1UuiZKQAAgLWdMAUAAFCAMAUAAFBArcPUl770pep///73v18pxQAAAKwuah2mHnjggXz88cdJkqOPPnqlFQQAALA6qPVqfl/4whdy7rnnZpdddkmS3Hvvvcsc+5+zWAAAAGuiWoepa6+9NqeddlquvPLKJMlXv/rVZY6dN2/e568MAACgHqt1mNpzzz3z4osvJkkqKioEJgAAYK1WaDW/sWPH1nUdAAAAq5Vaz0z9p2222SZJ8v7772fcuHEplUrp0qVLWrVqVafFAQAA1FeFZqbmzp2bE088Ma1bt07fvn3Tr1+/tG7dOieeeGLmzp1b1zUCAADUO4XC1ODBg/PII49k+PDhmTx5ct57770MHz48Dz/8cAYPHlzXNQIAANQ7hW7zu/322/OXv/wlO+ywQ3XbwIED0759++y33365+uqr66xAAACA+qjQzNSsWbPSvn37Jdrbt2+fqqqqz10UAABAfVcoTPXs2TOXXHJJSqVSdVupVMrPfvaz9OrVq86KAwAAqK8K3eZ32WWXZf/998+IESPSu3fvJMmzzz6bKVOm5IEHHqjTAgEAAOqjQjNTffv2zfjx43PkkUdm4cKFWbRoUY488siMHz8+ffv2resaAQAA6p1CM1NJ0rp165x//vl1WAoAAMDqo9DMFAAAwNpOmAIAAChAmAIAACigUJg67LDDCvUBAACsKQqFqdtvv32p7aVSKXfcccfnKggAAGB1sEKr+X344YdL/XeSLF68OE8++WTatGlTF3UBAADUaysUpjbccMOl/vtT66yzTi6++OLPXxUA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gDVcvQ5TG2+8cbbccsuMGjWqRvvjjz9evWhFbcYsTZMmTdK8efMaHwAAgNqq12EqSU499dTccMMNefLJJ7NgwYIMHTo07733Xk488cQVGgMAAFCXGpbz4H/84x9z8sknVz/b9KUvfSkNGzbM2WefnbPPPjtJ8t3vfjfTp0/PAQcckNmzZ6dDhw4ZOXJkOnfuXL2f2owBAACoSxWl5a3SsJItWLAgc+bMWaJ93XXXzbrrrlujrVQqZd68eWnatOky91ebMctSVVWVFi1aZObMmW75W4tN+Om25S6BZWh37thyl1Cneg3+bblLYBlGX3pMuUtgLeHvQP111waXlrsElmNV/DdBbbNBWWemGjdunMaNG9dqbEVFxWeGpNqMAQAAqAv1/pkpAACA+kiYAgAAKECYAgAAKECYAgAAKECYAgAAKECYAgAAKECYAgAAKECYAgAAKECYAgAAKECYAgAAKECYAgAAKECYAgAAKECYAgAAKKBhuQsAgPpkwk+3LXcJLEO7c8eWuwSAGsxMAQAAFCBMAQAAFCBMAQAAFCBMAQAAFCBMAQAAFCBMAQAAFCBMAQAAFCBMAQAAFCBMAQAAFCBMAQAAFNCw3AWsbXoN/m25S2A57tqg3BUAALC6MDMFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQgDAFAABQQMNyF/BZZs+enblz59Zoa9iwYVq2bLnE2MWLF2fOnDlZf/31V1F1AADA2qrez0x9//vfz+abb56uXbtWfwYMGFBjTKlUyo9+9KO0bNkyG220UTp27Jj77ruvTBUDAABrg3ofppJkwIABmT59evVn1KhRNfqvvPLKXHPNNfnLX/6SOXPm5KSTTspBBx2UcePGlaliAABgTbdahKnkk9v9Fi9evNS+q666KieccEJ22WWXNGzYMD/4wQ/Stm3bXH/99au4SgAAYG2xWoSpu+++OxtvvHHWW2+97L333hk7dmx137Rp0/Lmm29m1113rbHN7rvvnmeeeWZVlwoAAKwl6n2Y6tGjRx577LHMnj07EyZMyMYbb5x99tkn06ZNS5Lq/7vJJpvU2G6TTTap7lua+fPnp6qqqsYHAACgtup9mPr2t7+dPn36ZJ111skmm2ySm2++ObNnz86wYcOSJBUVFUmSjz/+uMZ2ixYtyjrrLPv0hg4dmhYtWlR/KisrV95JAAAAa5x6H6b+23rrrZe2bdvmzTffTJK0bds2STJlypQa46ZMmZI2bdoscz/nnHNOZs6cWf2ZOHHiyisaAABY46x2YWrq1KmZMGFCOnTokCRp0aJFtttuuzz00EPVYz7++OM88sgj2W233Za5nyZNmqR58+Y1PgAAALVVr8PU/Pnzs8cee+T+++/P22+/nVGjRmXgwIFp3bp1jjrqqOpxP/rRj3LzzTfnlltuybhx4/Ltb387ixYtykknnVTG6gEAgDVZw3IXsDxNmjTJJZdckosvvjgvvPBCNtxww+y2227505/+lJYtW1aPO+SQQzJv3rxcccUV+eEPf5htt902jzzySDbbbLPyFQ8AAKzR6nWYSpKdd945d95552eOO+aYY3LMMcesgooAAADq+W1+AAAA9ZUwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUIAwBQAAUMAaFaYmTpyY5557LlVVVeUuBQAAWMOtEWFq3rx5Ofjgg9OlS5ccffTR2WyzzXLVVVeVuywAAGAN1rDcBdSFIUOG5B//+EfeeOONtGnTJiNHjsygQYOy0047Zeeddy53eQAAwBpojZiZuvnmm3PCCSekTZs2SZKBAwdmm222yc0331zmygAAgDXVaj8z9e6772bKlCnp1atXjfaddtopL7zwwjK3mz9/fubPn1/9febMmUmy0p+3+nj+3JW6fz6fWY0+LncJLMOa9iykvwX1l78D9Ze/A6wq/g7Ub6vib8GnxyiVSssdt9qHqRkzZiRJWrVqVaO9VatW1X1LM3To0AwZMmSJ9srKyrotkNXKNuUugGUb2qLcFbCW8HegHvN3gFXE34F6bhX+LZg1a1ZatFj28Vb7MNWoUaMknyxC8Z/mzp2bxo0bL3O7c845J2eccUb198WLF2fGjBlp1apVKioqVk6x1GtVVVWprKzMxIkT07x583KXA5SBvwOAvwMkn8xIzZo1K23btl3uuNU+TFVWVmadddbJO++8U6P9nXfeSbt27Za5XZMmTdKkSZMabS1btlwZJbKaad68uT+esJbzdwDwd4DlzUh9arVfgKJZs2bp27dv7r777uq22bNn56GHHkr//v3LWBkAALAmW+1nppLkwgsvTP/+/XPOOeekT58+ueqqq7LpppvmW9/6VrlLAwAA1lCr/cxUkuyxxx7529/+lrfffjtXXnlltt566zzxxBNZf/31y10aq5EmTZrkvPPOW+L2T2Dt4e8A4O8AK6Ki9Fnr/QEAALCENWJmCgAAYFUTpgAAAAoQpgAAAAoQpiDJwoUL8/TTT2fcuHHlLgUokzlz5uTFF19c4r2FwNplwoQJGTNmTKqqqspdCqsBYYq12qxZs/KTn/wkW265Zfr375+f/vSn5S4JWMWmTJmS448/Pm3bts1xxx2X7bbbLn379s2bb75Z7tKAVeivf/1rtttuu+y999456qijstlmm+WMM86ItdpYHmGKtdq0adPSuHHj/OMf/0i/fv3KXQ5QBhMnTszuu++e6dOn54UXXsiECRPStGnTHHXUUeUuDViFJk2alLvuuiuvv/56xo4dm8cffzxXXnllRowYUe7SqMcsjQ7/z/7775+NN944v//978tdClBmt9xyS44//vjMnz8/DRuuEe+3B1ZQqVTKRhttlPPPPz+nnnpqucuhnvK/EADwX5599tm0b99ekIK1zKxZszJmzJh89NFH+eMf/5i2bdvmyCOPLHdZ1GP+VwIA/sOjjz6a66+/Pr/+9a/LXQqwik2cODFnn312ZsyYkUmTJuWiiy7KxhtvXO6yqMeEKQD4f0aPHp2BAwfm1FNPzXHHHVfucoBVrHv37nniiSeSfDJDvddee6VRo0Y58cQTy1wZ9ZUFKAAgyfPPP5/+/fvnuOOOy2WXXVbucoAy6927d/baa6/8+c9/Lncp1GPCFABrvRdffDH9+/fP0UcfnV/84hflLgcog9mzZ9f4vnjx4vz73/9Oq1atylQRqwO3+bHWe/LJJ1MqlfLhhx+mVCrliSeeSJMmTdK7d+9ylwasAuPGjcu+++6bbbbZJl/72teqb/FJkp122imNGzcuY3XAqrLLLrvkmGOOybbbbpvZs2fnN7/5TSZOnJjbb7+93KVRj1kanbXennvumUWLFtVo23TTTXPnnXeWqSJgVfrLX/6yzBd233XXXdlkk01WcUVAOUyfPj1XXXVVnn322ay77rrZbrvtctJJJ6V169blLo16TJgCAAAowDNTAAAABQhTAAAABQhTAAAABQhTAAAABQhTAAAABQhTAAAABQhTAAAABQhTAFBP3H333fn3v/9d7jIAqCUv7QVghXzwwQd58MEHkyQVFRVp1apVttlmm2y22WZlrqz+mzt3bp577rm8//772WKLLdK9e/c0a9asun+LLbbIhRdemGOPPbZ8RQJQaw3LXQAAq5c33ngjhx9+ePbff/+0aNEi7777bp555pmcf/75Oeecc8pdXr112WWX5YILLkjHjh3TsWPHvPvuu5k8eXLOOOOMnHbaaeUuD4AChCkACrn00kuzzTbbJEmuueaafPe7383AgQMzceLEzJgxIxUVFWnbtm169OiRDTbYYIntR48enXfeeSddunRJly5dat23YMGC/P3vf8+sWbPSvXv3bLnlljX677zzzuy0005p2LBhXnzxxTRv3jy77LJL1lmn5p3tkydPzj/+8Y9svvnm2X777fPXv/41Xbp0SadOnVb4WIsXL87zzz+fDh06pEePHkuc64UXXpiLLrooI0eOzBe/+MXq9pkzZ+buu+9e5jX+y1/+slKvJQCfjzAFwOc2cODAfOc738lLL72Ul156KW+88UZKpVJef/31vPPOO7nzzjvTt2/fJMm8efOy33775a233soOO+yQ119/Pd26dcuwYcOW25ckzz//fAYOHJhNNtkkbdu2zd///vcceuihueaaa6prOf7449OnT5+MHz8+3bp1y3PPPZeuXbvmoYceSoMGDZIkw4cPz9FHH50ddtghDRo0yJw5c/LOO+/k/PPPrw5TtT3WzjvvnNdeey09e/bMwQcfvESY+uCDD3LRRRfl9NNPrxGkkqRFixY5+uijl3ldH3vssZV2LQGoAyUAWAHPPvtsKUlp7Nix1W1/+9vfSklKjz322BLjhwwZUurRo0f195EjR5Y23njj0uzZs6vbRowY8Zl98+bNK1VWVpauuuqq6r733nuv1Lp169KwYcOq21q0aFHq06dPac6cOaVSqVSaPHlyqWnTpqW77767VCqVSh999FFpk002KV1wwQXV21x66aWlJKXrrrtuhY+1/fbbl2bNmrXM6zVy5MhSktJzzz23zDGf2nzzzUs333zzMvvr6loCUDfMTAFQyAMPPJCXX345kydPzuWXX55+/fqlX79+SZK33347r776aj788MM0adIkY8aMyfz589OkSZM0bdo0c+fOzbhx46pncQ466KAkWW7fww8/nHfeeScbb7xxhg8fnlKplFKplC233DJ/+9vf8tWvfrW6tmOPPTZNmzZNkmy22Wbp0qVLXnvttQwYMCCPP/54ZsyYkTPOOKN6/He/+938+Mc/rv6+Isc6/vjjs/766y/zOr333ntJknbt2hW6zivjWgJQN4QpAAp5+OGH07Jly2y44YY577zzcswxx6RBgwY5+eST89vf/jY77bRTWrVqlTlz5qRUKmX69OnZfPPN079//5x88snZY4890qZNm+yzzz458cQTs9122y2376233krjxo0zcuTIGnW0a9cunTt3rtG20UYb1fjepEmTzJs3L0kyceLEbLLJJjVW0WvSpEnatGlT/X1FjvWf2y3Np884vf/++9lkk02WO/a/raxrCUDdEKYAKOQ/F6D41D/+8Y/8+te/zuuvv54OHTokSZ544oncf//9Kf2/N3FUVFTkkksuyf/+7//mueeey+9///vsuOOOeemll9K1a9dl9jVv3jwLFizITTfdVCMIraiNNtooM2fOXKL9gw8+qP73ihyroqJiuf29e/dOkjz99NPp2rVrretcmddyReoAYNm8tBeAOvPee++lWbNmqaysrG4bPnz4EmMWL16cRo0apU+fPrnmmmvStGnTPP/888vt23vvvdOgQYPccMMNNfa3aNGiTJkypdY17rTTTlm4cGH+8pe/VLc99thjNQJWXR0rSTp37pwBAwZkyJAhNQLbpyZMmLDU7VbmtQSgbpiZAqDO9OvXL02aNMmhhx6aAw44IH//+98zYsSIGmMef/zxDB06NAcffHAqKyvz0EMPpXHjxtlzzz2X29e2bdv84he/yGmnnZZXXnklvXv3zsSJEzNixIhceuml2X///WtVY7t27XLKKafkiCOOyFlnnZV11lknV111VdZbb73qWaa6Otanbr755gwYMCDbbbddTjjhhOr3TD311FNp0KBB7rrrrlV6LQGoG2amAFghG220UQ499NC0bNlyib5WrVrl6aefzpZbbpnHHnssHTt2zKOPPppDDz20+na5Qw45JL/73e8yf/78PPbYY+nSpUtefPHFtG3bdrl9SXLKKafkmWeeSYsWLTJq1Kg0aNAgd955Z41w82l4+E/9+/fP1ltvXf398ssvzyWXXJJXXnklM2bMyL333pumTZvWeIdT0WMtTatWrfLEE0/kuuuuy8yZM/Pwww9n1qxZOemkk2oEqa985Svp2LHjKrmWAHx+FaVPb7wGgLXEjBkzaixS8dxzz6V3797517/+5XkiAGrNbX4ArHX++Mc/5oknnsjee++dadOm5corr8zRRx8tSAGwQsxMAbBWGjFiRB555JE0atQou+66a413RwFAbQhTAAAABViAAgAAoABhCgAAoABhCgAAoABhCgAAoABhCgAAoABhCgAAoABhCgAAoABhCgAAoABhCgAAoID/D77ueOgmu+69AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1000x800 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (10,8))\n",
    "sns.countplot(data = df, x= 'Pclass' , hue = 'Survived')\n",
    "plt.title('count of survive per class')\n",
    "plt.xlabel('Passenger Class')\n",
    "plt.ylabel('Count of passenger')\n",
    "# plt.legend(title = 'Survived', labels = ['No','Yes'])\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "d275ea6f-b204-4b50-a3a5-d69604af7a43",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pclass\n",
       "1    0.629630\n",
       "2    0.472826\n",
       "3    0.242363\n",
       "Name: Survived, dtype: float64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "survive_rate_per_class = df.groupby('Pclass')['Survived'].mean()\n",
    "survive_rate_per_class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "750c171f-b154-4f8a-ba62-1ce65de01b05",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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4IyF/JZW3759a1yofPToUa1atUp5eXm69dZbQ3q8H374Qd98842cTqfS0tIUFRVVY8bn8+mLL75Q48aNdcUVV5zxDAAAiEwhB8+ll16qSy+9tMb++++/X9dff73Ky8uDehyPx6OJEydqxYoVat++vfbv3y+n06nFixcHXCO0bt06DRs2TA6HQ16vV4mJiVq9enXAJzoHMwMAACJXrV7D07t376A/n+f7779Xnz59VFJSoi1btujAgQO67LLLNHr0aP+M2+3WkCFDNGbMGOXn56uwsFBJSUkaMWJESDMAACCy1VrwVFZWasOGDWratGlQ8+3atVNGRoYaNWokSYqJiVHPnj1VUlLin8nNzZXX69XEiRP9M1lZWfrggw+0e/fuoGcAAEBkC/ktrddee00vvvhiwL7jx48rLy9PjRo10vDhw0N6vO3bt+v7779XXl6eZs+erWeeecZ/zOVyKS0tTU6n07/vl8/5cblcuuiii4KaOZHP55PP5/Pf9ng8Ia0ZAAA0LCEHT5s2bfR///d/gQ8SE6O7775bd955pxwOR0iPt2DBAn388cfKy8tTp06d1K9fP/+x8vJytWjRImC+efPmio6O9l8rFMzMiXJycvT000+HtE4AANBwhRw811xzja655ppaW8DMmTMl/fyW2H333acbbrhBu3btks1mk81mU0VFRcB8ZWWlqqurZbPZJCmomRNlZ2drwoQJ/tsej0fJycm19pwAAED9ctbX8NTW20E2m01jx47Vvn37tGvXLklSSkqKioqKAuYOHDjgPxbszInsdrvi4+MDNgAAYK4zCp6CggJlZmYqPj5eDodDTZs21e233668vLygH8Pr9dbYt2vXLkVFRSkxMVGS1L9/f5WWlmrLli3+mZUrV6pp06ZKT08PegYAAES2kN/S+vHHH9W7d29dcMEFeu6555ScnKySkhK9/PLLuuaaa7Rjxw6df/75p32cefPm6aOPPtItt9yixMREff7555o+fboeeOABJSUlSZK6d++uu+66SxkZGZo6darKyso0efJkTZ06VXFxcUHPAACAyBZy8KxZs0bx8fHauHFjwDUyGRkZuuGGG/Taa69p7Nixp32cRx55RB07dtTy5ctVUlKiNm3aaOnSpbrxxhsD5hYtWqQ5c+Zo8eLFstvtmj9/voYOHRryDAAAiFwhB09xcbF69OhR44Lg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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sex\n",
      "male      577\n",
      "female    314\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "sns.countplot(data = df , x = 'Sex')\n",
    "plt.show()\n",
    "print(df['Sex'].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "ec171202-e05b-4791-a111-3482ae2561ea",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABEIAAAHqCAYAAAAAtetdAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlcelbwAAAAlwSFlzAAAPYQAAD2EBqD+naQAARTBJREFUeJzt3Xm0XvO9P/D3yeBkPIc0kUSmCmJqiqARcwlRMYSK0hhrrHsN7eWStrhUuaVcVRT3VlVRpIaoWZAS80yQmFKSEAnhnIwnkZzfH12eX48gOSR5crJfr7WetTzf/dl7f3bWbr/J+9lDRX19fX0AAAAACqBZuRsAAAAAWF4EIQAAAEBhCEIAAACAwhCEAAAAAIUhCAEAAAAKQxACAAAAFIYgBAAAACgMQQgAAABQGIIQAAAAoDAEIcBSc/vtt6dVq1bLfD+jR49ORUXFMt8PACwvc+fOTUVFRcaMGbPM99WiRYuMGjVqme8HYEUlCIGVzA033JDNN988VVVV6datW/bbb78899xz5W6ryXnzzTfzve99L23atEmnTp1ywgknZN68eeVuC4Bl6Nlnn83gwYOz+uqrZ9VVV813v/vd3HLLLamvry93a03KW2+9lZNPPjmdOnVKRUVFPvnkk3K3BNCAIARWIrfddlsOOOCAHHXUUZk4cWKef/75DB06NGecccZy2f9uu+2WuXPnLpd9LUt1dXUZNGhQ2rRpkzfffDP33HNP/vrXv+Y//uM/yt0aAMvIu+++mx133DG9evXKc889l0mTJuWXv/xlrrvuurz++uvLfP+tWrVKfX19tt5662W+r2XtpJNOymqrrZZTTz213K0AfC5BCKxEbrnllgwYMCCHH354qqur06lTp3z/+9/PrbfeWqq5+OKLs9566zVYb9SoUQ1uNbn++uvTsWPHXHLJJenZs2eaN2+eF154IW3bts0tt9zSYN0HHnggLVq0yJQpUxrcGjNz5szF1ifJ008/nR133DFt2rRJ165dc+ihh+bDDz9ssM6NN96YtdZaK+3bt8+WW26Zp5566mv/WX2ZW265JRMmTMhll12Wrl27pl+/fjn11FPzv//7v6mtrV2m+wagPB588MHU1NTkggsuSLdu3dKuXbtsvfXWGTFiRPr06ZMkGTduXCoqKjJp0qQG61ZUVJRuNfnggw9SUVGR3/3ud9l4442zyiqr5LLLLst3v/vd/Nu//VuD9WbPnp327dvnL3/5yyK3xiyuPkmmTZuWgw8+ON/4xjdSXV2dHXfcMS+88EKDdcaOHZstt9wyrVu3zgYbbJArr7xy6f2hfYGbbropp5xySlZfffVlvi+Ar0IQAiuRDh06ZPz48Xn77be/9rY+/PDD3H777Xn44YezYMGCbLTRRtljjz1y7bXXNqi79tprs+OOO6ZLly4Nxtu1a7fY+tdeey077rhjhg4dmvfeey9PP/10Pv744+y3336l+pdeeik//OEP8x//8R+ZNGlSTjvttJx11llf2vs//vGPVFRUfOnngAMO+ML1H3300Wy44Ybp1KlTaWzHHXdMXV1dnnnmmcX+2QHQ9HTo0CH19fW57777lsr2Lrjgglx22WWZM2dOjj766BxwwAG58cYbG9wmMnLkyCTJnnvuucj6i6ufP39+Bg4cmJYtW+bFF1/MxIkTM3DgwAwcODAffPBBkmTevHnZbbfdss466+Ttt9/OzTffnN/+9rdZsGDBl/a+9tprf+kc+tk5H6CpEYTASuTEE09Mly5dsvbaa2fAgAE54YQTcuedd2bhwoVfaXtXXHFFevXqVfo+bNiw3H777ampqUnyz1tIbrrppi8MFRZXf/755+d73/tejj766FRXV6dbt2655JJLMmrUqFKYc+GFF2aHHXbIMccck+rq6uyyyy459thjv7Tvb37zm6mvr//SzzXXXPOF60+ZMqVBCJKk9KvWp1eyALBy2WWXXTJs2LDsscceWX/99XPooYfm6quv/spXAp522mnZYost0rx58yTJPvvskxkzZuSee+4p1Vx77bXZe++906ZNm0XWX1z9rbfemilTpuSKK65It27dUlVVleHDh2f11VcvBSY33XRTpk+fnksuuSSrr7561ltvvfz2t79dbO9vvPHGl86h5kKgqROEwEqka9euefbZZzNq1KgMGjQor7zySvbYY49sv/32jX7QZ/v27dOjR48GY7vsskvat2+fm266Kck/3xIzf/787LXXXp+7jcXVP/3007nxxhvTokWLNG/ePM2aNUu3bt2S/PNBa0nyyiuvZLPNNmuw3c0337xRx7I0fPqgPG+rAVg5VVRU5JprrsnLL7+cI488MnPmzMlxxx2X9ddfP+PHj2/09jbccMMG36urq7PbbruVrpT84IMPcu+9937hjwmLq3/66aczderUrLLKKmnevHlpHn3llVcazKHrrbde2rVrV9puOeZQgBWNIARWMs2bN892222X//qv/8q9996bBx54IA8//HBuuOGGL1zn864YWWWVVRYZa9GiRfbdd9/SX8quvfbaDBkypMFfsBpTv3Dhwpxwwgn55JNPsmDBgixcuLD0a9N3v/vd0nYaGz583VtjunTpkmnTpjUY+/R7586dG9ULAE3LBhtskJ/85Ce5/vrrSw9JPfvss7+w/ouuuvy8eXTYsGEZOXJkZs6cmRtvvDEdO3bMDjvs8IXb/rL6hQsXpm/fvqU59F/n0V/96lelbXyVAN+tMcDKThACK7ktt9wyLVq0yEcffZQkWW211Ur//anGPA3/gAMOyOjRo/Pyyy/nzjvv/NJAYXH1/fr1y7333vult+5ssMEGizwcdXEPS/26t8ZsueWWefnllzN16tTS2AMPPJDKyspsuummX7pvAFYenTp1yvrrr99gDk3SYB5tzBw6ePDgVFZW5tZbb821116b/fffv3TrTGPr+/Xrl1dfffVLnwu2wQYb5NVXX82sWbNKY0vywHG3xgArO0EIrEROOumknH322Xn55Zczd+7cTJo0Kccff3xatmyZnXfeOUkyYMCATJ8+PVdccUVmz56dMWPG5JxzzlnifQwYMCBrrrlmhg0blurq6uy0005fuf6kk07KP/7xjxxxxBF5++23U1NTkwceeCCDBw8u1Rx33HF54IEH8vvf/z61tbW5++6787vf/a6RfzKNs9dee2XNNdfMj3/840yZMiXPPvtszjzzzBxxxBGpqqpapvsGoDxuuOGGHHXUUXnsscdSW1ubjz76KH/4wx/y0EMPlR5m2rlz5/Tp0yfnnXdeampq8uabb+aYY45Z4n2sssoq2WeffXLeeefl0UcfXeyPCV9W//3vfz/rr79+hg4dmmeffTazZs3KCy+8kB//+Md5/PHHkyR77713VltttRxzzDGZNm1axo0bl+OPP/4r/OkArFwEIbASOfHEEzN37tzsv//+6dChQzbbbLO89dZbueeee0qvzO3du3f+8Ic/5L//+7/TpUuX/OIXv8gJJ5zQqP388Ic/zAsvvJD99tsvLVq0+Mr16623XsaMGZP33nsvffv2zTe/+c2cc845Oemkk0o1G220Ua655pqce+65WWONNXLmmWfm5z//eaP6bazKysrcc889mTVrVnr37p2dd9453//+93P++ecv0/0CUD577rlnNt9885x00knp2bNnevfuncsuuyyXXnppDjvssFLdddddl1dffTVrrLFGhgwZ0mDZkjjggAPy4osvZv31188mm2zyletXWWWVPPjgg9l4442z6667plOnTjn00EOzySablJ4DUllZmb/97W8ZP358evbsmb322ivHHXfcl16FsjQcfvjhqaioyP77758kadmyZYNXDAOUW0X9p08ABAAAAFjJuSIEAAAAKAxBCAAAAFAYghAAAACgMAQhAAAAQGEIQgAAAIDCEIQAAAAAhSEISVJfX5/a2tp4kzAANI45FABoagQhSWbMmJHq6urMmDGj3K0AQJNiDgUAmhpBCAAAAFAYghAAAACgMAQhAAAAQGEIQgAAAIDCEIQAAAAAhSEIAQAAAApDEAIAAAAUhiAEAAAAKAxBCAAAAFAYghAAAACgMAQhAAAAQGEIQgAAAIDCEIQAAAAAhSEIAQAAAAqjbEHIwoUL85e//CUDBgxIu3bt0r179xx11FH58MMPG9RNnDgxe+yxR9q2bZtvfOMbOeaYYzJnzpxG1wAAAACULQh5+eWX8+CDD+biiy/OtGnTctddd+WJJ57IgQceWKr55JNPsuuuu+aTTz7Ja6+9ltGjR+fuu+/OMccc06gaAAAAgCSpqK+vry93E5+64IIL8stf/jIfffRRkuT222/P7rvvnn/84x/p1atXkuTaa6/NwQcfnHfffTerr776EtUsTm1tbaqrq1NTU5Oqqqpld4AAsJIxhwIATc0K8YyQhQsXZty4cbnuuusydOjQ0vijjz6aNddcsxRwJMmOO+6YBQsW5IknnljiGgAAAIBkBQhCBg4cmObNm2f99ddPx44d89vf/ra0bMqUKenUqVOD+k6dOqWioiJTpkxZ4prPqqurS21tbYMPALB45lAAoKlrUe4GRo0alfnz52fs2LE54ogjsscee+S+++5b7HoVFRVfueacc87JGWec0eheAVixvXNm33K3sFz1PO2l5b5PcygA0NSV/YqQJGnZsmU22WST/OY3v8moUaPy2muvJUm6dOmSadOmNaidNm1a6uvr07lz5yWu+azhw4enpqam9Jk4ceIyOCoAWPmYQwGApm6FCEI+NX/+/CT//0qOLbfcMhMmTMjbb79dqnnggQfSvHnz9O/ff4lrPquysjJVVVUNPgDA4plDAYCmrmxvjbn44ovTvHnzDB48OB07dszzzz+fo446KtXV1RkzZkySf74ad5NNNkmPHj3yv//7v5k+fXr23HPPbLfddvnjH/+4xDWL44n3ACsHt8Ysf+ZQAKCpKdsVIcOGDcuECRPy3e9+N506dcqhhx6a3XbbLbfffnuppkWLFrnjjjvSvHnz9OnTJ9ttt10GDRqUSy+9tFE1AAAAAEkZrwhZkfg1C2Dl4IqQ5c8cCgA0NSvUM0IAAAAAliVBCAAAAFAYghAAAACgMAQhAAAAQGEIQgAAAIDCEIQAAAAAhSEIAQAAAApDEAIAAAAUhiAEAAAAKAxBCAAAAFAYghAAAACgMAQhAAAAQGEIQgAAAIDCEIQAAAAAhSEIAQAAAApDEAIAAAAUhiAEAAAAKAxBCAAAAFAYghAAAACgMAQhAAAAQGEIQgAAAIDCEIQAAAAAhSEIAQAAAApDEAIAAAAUhiAEAAAAKAxBCAAAAFAYghAAAACgMAQhAAAAQGEIQgAAAIDCEIQAAAAAhSEIAQAAAApDEAIAAAAUhiAEAAAAKAxBCAAAAFAYghAAAACgMAQhAAAAQGEIQgAAAIDCEIQAAAAAhSEIAQAAAApDEAIAAAAUhiAEAAAAKAxBCAAAAFAYghAAAACgMAQhAAAAQGEIQgAAAIDCEIQAAAAAhSEIAQAAAApDEAIAAAAUhiAEAAAAKAxBCAAAAFAYghAAAACgMAQhAAAAQGEIQgAAAIDCaFHOnS9cuDDPPfdc3nrrrfTo0SPf+c530qzZ/89mPvnkk1xzzTWLrLf11ltn7bXXbjD23nvvZcyYMWnVqlW23377tG/ffpn3DwAAADQtZQtCHn300Rx55JFp1apV1lxzzTz11FOprq7OXXfdlTXWWCNJMnfu3Bx66KHZdddd06lTp9K6a6+9doMg5Nprr82RRx6ZAQMG5KOPPsqkSZNy1113pV+/fsv9uAAAAIAVV9mCkAULFmTEiBFZf/31k/wz9Nhyyy3zk5/8JDfccEOD2lNPPTVbbLHF525n6tSpOfLII3P22Wfn+OOPT5L84Ac/yMEHH5yXXnpp2R4EAAAA0KSU7Rkh22yzTSkESZJWrVpl4MCBnxtePP7447nhhhvy9NNPZ+HChQ2W3Xrrramvr88RRxxRGjvuuOMyduxYQQgAAADQQFmfEfKvFi5cmPvuuy8bbbRRg/GKiopcf/316dq1ax599NH06tUrf/3rX9OzZ88kydixY7PmmmumTZs2pXW+9a1vlZb17dt3kX3V1dWlrq6u9L22tnZZHBIArHTMoQBAU7fCvDXmF7/4RV5//fX813/9V2msZcuWefjhh/P444/nlltuyeuvv55PPvmkwdUfNTU1WXXVVRtsq7q6Os2bN09NTc3n7uucc85JdXV16dOjR49lcUgAsNIxhwIATd0KEYRccMEF+Z//+Z/cfPPNWXfddUvjlZWV2WqrrUrfq6qqcvzxx+f+++8v/RrVunXrzJo1q8H25syZkwULFqR169afu7/hw4enpqam9Jk4ceIyOCoAWPmYQwGApq7st8ZceOGF+fnPf55bbrklO++882Lr27RpkwULFuTjjz9O586ds9Zaa2XEiBFZuHBh6dW7EyZMSJKstdZan7uNysrKVFZWLr2DAICCMIcCAE1dWa8IueiiizJ8+PDcfPPN2WWXXRZZPmnSpNTX1zcYGzFiRNZcc8107tw5STJ48OB89NFHue+++0o1f/nLX9KpU6f0799/2R4AAAAA0KSU7YqQG264Iccff3z233//vP/++7nqqquSJKusskp++MMfJkkefPDBXHLJJdl1112z6qqr5s4778yjjz6aG2+8sbSdDTbYIP/+7/+eAw44ICeccEKmT5+eiy66KFdddVVatmxZjkMDAAAAVlBlC0KaNWuWgw8+OEkyevTo0nibNm1KQciBBx6YTTbZJDfffHNef/31DBo0KH/6059KV4N86qKLLsp2222XBx54IJWVlXn44YezxRZbLLdjAQAAAJqGivrP3ntSQLW1tamurk5NTU2qqqrK3Q4AX9E7Zy76yvSVWc/TXip3C+ZQAKDJWSHeGgMAAACwPAhCAAAAgMIQhAAAAACFIQgBAAAACkMQAgAAABSGIAQAAAAoDEEIAAAAUBiCEAAAAKAwBCEAAABAYQhCAAAAgMIQhAAAAACFIQgBAAAACkMQAgAAABSGIAQAAAAoDEEIAAAAUBiCEAAAAKAwBCEAAABAYQhCAAAAgMIQhAAAAACFIQgBAAAACkMQAgAAABSGIAQAAAAoDEEIAAAAUBiCEAAAAKAwBCEAAABAYQhCAAAAgMIQhAAAAACFIQgBAAAACkMQAgAAABSGIAQAAAAoDEEIAAAAUBiCEAAAAKAwBCEAAABAYQhCAAAAgMIQhAAAAACFIQgBAAAACkMQAgAAABSGIAQAAAAoDEEIAAAAUBiCEAAAAKAwBCEAAABAYQhCAAAAgMIQhAAAAACFIQgBAAAACkMQAgAAABSGIAQAAAAoDEEIAAAAUBiCEAAAAKAwBCEAAABAYQhCAAAAgMIQhAAAAACFIQgBAAAACkMQAgAAABRGi3Lu/PHHH8+1116bt956Kz169Mjhhx+ezTbbrEHNvHnzcvHFF+f+++9Pq1at8oMf/CD77rtvo2sAAAAAynZFyOWXX56f/vSnWXfddXPMMcdk1VVXTf/+/XPrrbc2qBs2bFguueSS7L///tl+++1zyCGH5IILLmh0DQAAAEBFfX19fTl2/PHHH2fVVVdtMDZs2LC8/fbbGTNmTJLkqaeeyne+8508/vjj6d+/f5Lk/PPPzxlnnJH3338/rVu3XqKaxamtrU11dXVqampSVVW1dA8UgOXmnTP7lruF5arnaS+VuwVzKADQ5JTtipDPhiCfjs2ePbv0/d57703nzp1LAUeSDBkyJDNmzMhjjz22xDUAAAAASZmfEfKvJk+enOuuuy7HH398aeztt9/OGmus0aCue/fupWVLWvNZdXV1qaurK32vra39+gcAAAVgDgUAmroV4q0xM2bMyJ577pl11103w4cPL43PmzdvkVtbVllllTRr1izz5s1b4prPOuecc1JdXV369OjRYykfEQCsnMyhAEBTV/YgZObMmfne976XhQsX5q677kplZWVp2WqrrZYPP/ywQf3HH3+chQsXZrXVVlvims8aPnx4ampqSp+JEycu5aMCgJWTORQAaOrKGoTMnDkzu+66a2bNmpVRo0YtElxssskmeeutt/Lxxx+Xxp566qkkycYbb7zENZ9VWVmZqqqqBh8AYPHMoQBAU1e2IGT27NkZPHhwZs6cmfvvvz8dOnRYpGbPPfdM+/btc+655yZJPvnkk5x77rnZaqut0qdPnyWuAQAAAEjK+LDUc889Nw899FA22mij7L333qXxdu3a5fbbb0+SVFdX54Ybbsj++++fESNGZMaMGfnGN75RWr6kNQAAAABJGYOQgw46KNtvv/0i4y1aNGxp4MCBmTRpUl588cVUVlamb9++qaioaHQNAAAAQNmCkN69e6d3795LVFtZWZnNN9/8a9cAAAAAxVb2t8YAAAAALC+CEAAAAKAwBCEAAABAYQhCAAAAgMIQhAAAAACFIQgBAAAACkMQAgAAABSGIAQAAAAoDEEIAAAAUBiCEAAAAKAwBCEAAABAYQhCAAAAgMIQhAAAAACFIQgBAAAACkMQAgAAABSGIAQAAAAoDEEIAAAAUBiCEAAAAKAwBCEAAABAYQhCAAAAgMIQhAAAAACFIQgBAAAACkMQAgAAABSGIAQAAAAoDEEIAAAAUBiCEAAAAKAwBCEAAABAYTQ6CJk8eXL+8Y9/NHoZAAAAQLk1Ogj5y1/+kosvvrjRywAAAADKbaneGvPxxx+nqqpqaW4SAAAAYKlpsaSFd955Z6677rq8+uqrmTNnTqZMmdJg+axZs3Lffffl5ptvXupNAgAAACwNSxyENGvWLC1atEizZs1K//2vevTokT/96U/Zeeedl3qTAAAAAEvDEgchu+yyS3bZZZf8/e9/z4wZM7Lbbrsty74AAACApWT27NkZNWpUJk2alO7du2e77bZLdXV1udsqi0Y/I2S77bYTggAAAEAT8cwzz2TNNdfMWWedlbFjx+aqq67KJptskquuuqrcrZXFEl8R8q+ee+65XHzxxZkwYULmzZvXYNn++++ff/u3f1sqzQEAAABfz7HHHputt946N910U2lsxowZee655xrULViwIPfdd1/eeOONdO/ePQMHDky7du2SJG+++WZuueWWHHLIIenYsWOSf15lctlll2XgwIH59re/vfwO6GtqdBDy9ttvZ5tttsk222yTAQMGpGXLlg2Wr7POOkutOQAAAODrmTBhwiLP82zfvn223Xbb0vcPPvgggwYNSn19fbbYYovcfPPNOf744zNq1Kiss846+eY3v5mbb745Dz/8cEaOHJkkOeGEE/LQQw/l6KOPXq7H83U1OggZNWpUttlmm9x1113Loh8AAABgKdppp51y0UUXpbq6OjvvvHPWW2+9NG/evEHNCSeckLXXXjs33HBDaezoo4/OT37yk9x+++1p3rx5rr322my00Ua57LLL0rlz5/zpT3/KY489ljZt2izvQ/paGh2EtGnTJj169FgWvQAAAABL2e9///v07Nkzv/vd7/LTn/40q666agYPHpyzzz47PXv2zPz583PTTTdl//33z4UXXpj6+vrU19dn7ty5eeyxx0rbWXPNNXPJJZfkqKOOSuvWrfPLX/4y/fr1K+ORfTWNfljqtttum4ceeigffvjhsugHAAAAWIratm2bs846K2+99Vbefffd/O53v8sjjzySgQMHZv78+Zk+fXrmzp2b2bNnZ9KkSZk8eXLefffddOzYMYceemiDbe27775p3759Pvnkkxx33HFlOqKvp9FXhLz88supqKjIeuutlx122CHt27dvsHzQoEEZOnToUmsQAAAAWDq6du2aAw44IB06dMjgwYMzbty4rLPOOmnRokUGDRq0SPDxWaecckratWuXysrKnHrqqTnvvPOWU+dLT6OvCJk3b1422mij7LjjjqmoqMjMmTMbfD77FhkAAACgfB5//PFFxt55550k/wxGWrVqlUGDBuWiiy5KXV1dg7oXX3yx9N/33ntvLrnkklxzzTX585//nAsvvDAPPPDAsm1+GWj0FSG77bZbdtttt2XRCwAAALCUnXvuuXnrrbey1VZbZY011sj48eNz44035he/+EXpVbiXXnppBg4cmI033jhDhgzJvHnzMnr06Hzve9/Lt7/97Xz44Yc55JBD8rOf/SwDBgxIkpx44ok5+OCD8+KLL2a11VYr5yE2SkV9fX19uZsot9ra2lRXV6empiZVVVXlbgeAr+idM/uWu4XlqudpL5W7BXMoADQRL7/8cv7+97/n3XffTefOnbPjjjtmgw02aFAzd+7c3HLLLXn55ZfTsWPH7LDDDvn2t7+dJLnrrrvy2GOP5bTTTkuLFv+8pmL+/Pk588wzs/XWW2fQoEHL/Zi+qkYHISNHjswf//jHL1w+ZMiQHHLIIV+3r+XKX+IAVg6CkOXPHAoANDWNvjWmXbt26d69e4OxWbNm5f7770/z5s2bXAgCAAAAFEejg5Add9wxO+644yLjs2fPzoABA9K7d++l0hgAAADA0tbot8Z8kTZt2mTPPffMqFGjltYmAQAAAJaqpRaEJMlbb73l9bkAAADACqvRt8Y8/PDDueOOOxqMLViwIC+99FJGjx6d4cOHL7XmAAAAAJamRgchU6ZMydNPP91wIy1apGfPnhkzZkw23HDDpdYcAAAAwNLU6CBk6NChGTp06LLoBQAAAGCZWqrPCAEAAABYkTX6ipAkmT9/fi677LL87W9/y6RJk9K1a9cMHDgwJ5xwQlq3bt3o7X3wwQdZuHBhVl999Qbj9fX1mTx58iL1HTp0SJs2bRYZ/+ijj7LKKqukbdu2je4BAAAAimrWrFmZOHFi2rVrl+7du5e9lwkTJuRb3/rWMtl+o68Iqa+vz6BBg3L66aend+/eGTZsWDbYYINcdNFFGTBgQOrq6pZ4WzfddFMGDhyYrl27Ztddd11k+axZs9KjR4/069cvW2yxRenz2Ye1jh07Nv369csaa6yR1VZbLXvssUemT5/e2EMDAACAwrnsssvSpUuX7LXXXrnkkkvK3U6eeuqp9O3bN5988sky2X6jrwi577778vrrr+fVV19N586dS+O/+tWvMmDAgNxwww056KCDFrudurq6XHfddfnP//zPbLjhhnnkkUe+sPa2227LFlts8bnL5s6dm9122y3bbrttHn300cycOTMDBw7Mj370o9x6662NPTwAAABYqjY96erltq9nzlv8v8c/69RTT81ll12WYcOGLYOOVjyNviJk3Lhx2WWXXRqEIElSVVWVvffeO+PHj1+i7VRWVuamm27KzjvvnIqKii+tnT9/fj788MPPXXb77bdn4sSJOffcc9OqVat07Ngxp59+em677bZMmjRpyQ4KAAAACmb27Nl57rnnSo+rGDt2bGbMmFFaXl9fnwkTJmTatGmLrDtjxoy88sorSZJ58+blrbfeanCHyMcff5wJEyZk4cKFDdabN29exo4dm7Fjx+a1117LvHnzlrjfL+unMRodhKy++up58cUXFzmYJHn22WfTqVOnr9XQ59lhhx2y5pprZtVVV83JJ5+cuXPnlpY9+eST6d27d7p06VIa22abbVJfX5+nnnpqqfcCAAAAK4Px48fngAMOSJKcfvrp2W+//fLcc88lSUaOHJmePXtmm222yQYbbJCNNtooL774Ymndv//979lkk01yxBFHpHv37tlyyy2zxhpr5J577slhhx2WPn36pH///llrrbXy2muvldabPHly9ttvv+y3337ZddddU1VVleOOO+5zM4Z/tbh+GqPRQcjgwYMzceLE7L777rntttvyzDPP5M4778y+++6bMWPGZN999/1KjXxuc82a5cwzz8yHH36Y2tra3H777fnjH/+Yk08+uVQzbdq0dOzYscF6HTp0SLNmzb4wJaqrq0ttbW2DDwCweOZQAFh5bLLJJnnmmWeSJFdffXXGjh2bbbfdNs8//3wOOOCAXHnllZk0aVKmTp2aPffcM9///vcbXMExb968VFZW5r333su7776b7bffPoMHD86qq66aKVOm5L333st6662X0047rbTOmmuuWboi5I033si4ceMycuTI/PnPf/7CPpe0nyXV6CCkffv2GT16dOrr67P33ntns802y+677573338/o0ePzhprrNHoJr5ImzZtcuqpp6aqqipJsvXWW2f48OH5v//7v1Ja1KxZs0UeoLJgwYIsXLgwzZs3/9ztnnPOOamuri59evTosdR6BoCVmTkUAFZ+l1xySbbccsv06NEjr776al599dUMGTIkb775ZsaOHdug9swzz0zz5s3TrFmz7Lnnnlm4cGHOPPPMNGvWLM2bN8/uu++e559/fpF9zJkzJ2+88UZmzpyZbbfdNqNGjVoq/SyJr/T63D59+uTOO+/M/Pnz8+6776ZLly6prKz8KptqtN69e2f27NmZOnVqunTpku7du+fee+9tUPP+++8nSbp16/a52xg+fHh++tOflr7X1tb6ixwALAFzKACs/F599dWMHz8+++yzT4PxDTbYIDU1NaXvlZWV6dChQ+l7mzZtUlVVlbZt2zYYmzlzZun7rFmzcsghh+S2225Lly5d0r59+0ydOvVLX5W7pP0sqa8UhHyqZcuW6dWr19fZxJdauHBhmjVreNHK448/nqqqqtLtMNtuu23OPPPMjBs3Luutt16S5O67784qq6zyhW+aqaysXG7BDQCsTMyhALDyW2WVVfK9730vV1+99N92c8455+S1117L5MmTS/+uP+644770eR9Lu59G3xpTX1+fvfbaK0888USD8ddffz077bRT5s+fv8Tbev/99zNp0qTMnDkz8+fPz6RJkzJp0qTU19cnSS688MKccsopefTRR/PKK6/kN7/5TS644IKccsopadHinxnODjvskK233joHH3xwnnjiidx111352c9+lmOPPTarrrpqYw8PAAAACm3rrbfO3XffvcizwL7K8zg+a9y4cdlmm21KIcjChQvz4IMPLtd+Gh2EjBo1KrNnz07//v0bjK+zzjrp2rVrRowYscTbGjZsWLbYYovcfffd+fDDD7PFFltkiy22yJw5c5Ikxx57bLp165ZTTjkl3//+9/Pggw9mxIgRGT58eGkbFRUVGTlyZPr165eDDjooJ598co4//vj8+te/buyhAQAAQOGdcMIJqaqqys4775zbbrstjz76aC699NJsvPHGWbBgwdfa9lZbbZUbb7wxt9xySx5++OEMGzYs48aNW679NPrWmPHjx3/h7TC9evXKq6++usTb+rKHoST/vPXm2GOPzbHHHvuldR06dMjvf//7Jd4vAAAA8M8XkGy44YYNnuvRoUOHPPnkkzn//PNz7rnnpr6+Pv369csdd9xReilJVVVVNtxwwwbbqq6uzgYbbNBgbLXVVis9xiL55wUPs2fPzm9+85vU19fnu9/9bs4666wGYUjbtm2z4YYbpqKiYon7aYyK+k/vQ1lCd955Z44//vi89NJLadWqVWl8wYIFGTBgQI488sgcfvjhjW6knGpra1NdXZ2amprSG2oAaHreObNvuVtYrnqe9lK5WzCHAgBNTqOvCNl5553TsmXLDBw4MD/5yU/Sq1evvPvuu7nkkksyefLk7LvvvsuiTwAAAICvrdHPCGnRokXuueeeVFVVZd99983mm2+eIUOGZO7cubn//vv9GgQAAACssL7S63N79OiRO++8MzNmzMjUqVPzjW98wxtaAAAAgBXeVwpCPtW+ffu0b99+afUCAAAAsEw1+tYYAAAAgKZKEAIAAAAUhiAEAAAAKAxBCAAAAFAYghAAAACgMAQhAAAAwFL32GOPpXv37lmwYEG5W2nga70+FwAAAPhy75zZd7ntq+dpLy23fS1OXV1dJk+enPr6+nK30oArQgAAAIDCEIQAAABAQd1///1Ze+2189e//jU777xz+vTpk3322SeTJ0/OjTfemK222irrrrtujjrqqMycObO03ptvvpnu3bune/fu6d27d3baaafccccdi93fxIkTc9hhh2XdddfNpptumpNPPjmzZs1aloe4CLfGAAAAQEHNmTMnb775Zi644IKcf/75qayszMEHH5ytttoqvXr1ygUXXJCKiooccMABOeOMM3LeeeclSXr16pXHH388STJ37tyMHj06++yzT+6///5sueWWn7uvadOmZcCAAfnBD36Qm266KXPnzs3w4cMzdOjQ3HnnncvtmAUhAAAAUHBXXXVV+vTpkyQ55phjcswxx+Shhx5Kz549kyRHHHFErrvuulJ9ixYt0r1799L3tddeO08++WT+9Kc/fWEQctFFF6VPnz45//zzS2PXXHNNunTpktdee620/2VNEAIAAAAF1rJlywYhRKdOndKuXbtS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XuRz8803N6irra3NT3/602y00Ubp379/LrjggixcuLBMXQMAAAArqhblbuDLLFy4MG+++WZGjBiRjTfeuDTeuXPnBnVDhgzJxx9/nN/+9rf56KOPcvjhh+eDDz7I2WefvZw7BgAAAFZkK3QQ8qnu3btn7bXX/txlo0ePzoMPPpixY8dmww03TJJMnTo1xx9/fE455ZRUVVUtz1YBAACAFdgKfWvMp4466qj07ds3Q4YMyV133dVg2ejRo9OtW7dSCJIku+66a+rq6vLYY48t71YBAACAFdgKf0XIwIEDc/zxx6dr16656667sueee+aSSy7JEUcckSSZOHFiunTp0mCdT79PmjTpc7dZV1eXurq60vfa2tpl1D0ArFzMoQBAU7dCByFt27bNvffem4qKiiTJpptumpqamvz85z8vBSELFy5My5YtG6zXvHnzNGvWLAsWLPjc7Z5zzjk544wzlm3zALASMocCAE3dCn1rTEVFRSkE+dTWW2+dadOmZdq0aUmSjh075sMPP2xQ89FHH2XhwoXp2LHj5253+PDhqampKX0mTpy4bA4AAFYy5lAAoKlboYOQz/OPf/wjLVq0SLt27ZIkm2++ed58881MnTq1VDNmzJgkyWabbfa526isrExVVVWDDwCweOZQAKCpW6GDkOuvvz5333136RaXp59+Ov/93/+dH/zgB2ndunWSZLfddkvXrl3zs5/9LJ988klqampy1llnZdddd03Pnj3L2T4AAACwglmhnxHSv3//nHjiidl3333TqlWrzJ49O0ceeWR++ctflmratGmTv/3tbxk2bFg6dOiQurq6bLvttrnqqqvK1zgAAEAT886ZfcvdwnLT87SXyt0CZbRCByFrrrlmbrrpptTV1eXjjz9O586dP7duk002ySuvvJL33nsvlZWV6dChw3LuFAAAAGgKVugg5FOVlZVfGIL8q65duy6HbgAAvhq/tgJA+a3QzwgBAAAAWJoEIQAAAEBhCEIAAACAwhCEAAAAAIUhCAEAAAAKQxACAAAAFIYgBAAAACgMQQgAAABQGIIQAAAAoDAEIQAAAEBhCEIAAACAwhCEAAAAAIUhCAEAAAAKQxACAAAAFIYgBAAAACgMQQgAAABQGIIQAAAAoDAEIQAAAEBhCEIAAACAwhCEAAAAAIUhCAEAAAAKQxACAAAAFIYgBAAAACgMQQgAAABQGIIQAAAAoDAEIQAAAEBhCEIAAACAwhCEAAAAAIUhCAEAAAAKQxACAAAAFIYgBAAAACgMQQgAAABQGIIQAAAAoDAEIQAAAEBhCEIAAACAwhCEAAAAAIXRotwNAAAAjffOmX3L3cJy1fO0l8rdArCScEUIAAAAUBiCEAAAAKAwBCEAAABAYQhCAAAAgMIQhAAAAACFIQgBAAAACkMQAgAAABSGIAQAAAAoDEEIAAAAUBiCEAAAAKAwBCEAAABAYQhCAAAAgMIQhAAAAACF0aLcDQCL2vSkq8vdwnJzS/tydwAAABSJK0IAAACAwlhpgpD58+fnueeey6uvvlruVgAAAIAV1Epxa8yDDz6Y/fffP61bt86MGTPSrVu33HbbbenVq1e5WwMAAABWIE0+CKmtrc3QoUNz2GGH5de//nXmz5+fQYMG5cADD8xDDz20XHrwPAcAAABoGpr8rTEjR45MbW1thg8fniRp2bJlTj755Dz88MN54403ytwdAAAAsCJp8leEPPfcc+ndu3dWXXXV0th3vvOd0rK11157kXXq6upSV1dX+l5TU5Pkn1eXfBUL6uZ8pfWaohktF5S7heXqq54TX5dzauXkfFr2inQ+JV//nGrfvn0qKioatY459Osp0jnq//OWvSKdT0l5zqkinU9Jsc6pcsyhrDgq6uvr68vdxNdxyCGH5PXXX88jjzxSGquvr0+LFi1y6aWX5qijjlpknf/6r//KGWecsTzbBIAVTk1NTaqqqhq1jjkUAL7aHMqKo8kHIUcccUSeffbZPPPMM6Wxurq6tGrVKldeeWUOPfTQRdb57K9ZCxcuzPTp0/ONb3xDqvclamtr06NHj0ycONH/6FkqnFMsTc6nxlsaV4SYQ5ecc5SlyfnE0uacahxXhDRtTf7WmF69euVvf/tbg7HJkycnSXr27Pm561RWVqaysrLB2L/eWsOXq6qq8n+OLFXOKZYm59OyZQ79+pyjLE3OJ5Y25xRF0OQflrrTTjvl/fffz5NPPlkaGzlyZNq1a5cBAwaUsTMAAABgRdPkrwjp379/9tprrwwbNixnnXVWpk+fnlNPPTWnn3562rRpU+72AAAAgBVIkw9CkuQvf/lLLrzwwlx55ZWprKzM5ZdfnmHDhpW7rZVOZWVlTj/99EUuiYavyjnF0uR8YkXnHGVpcj6xtDmnKJIm/7BUAAAAgCXV5J8RAgAAALCkBCEAAABAYQhCAAAAgMIQhLDEpkyZkjFjxqSmpqbcrbASmD59ep577rlMnz693K2wkpgwYUJeeumlzJw5s9ytwCLmzJmTRx99NG+//Xa5W2ElMGfOnLzwwguZOHFiuVthJTFt2rQ8++yz+eCDD8rdCiwXghAW69lnn83QoUOz0UYbZZtttslTTz1V7pZowl544YUMGjQo66yzTn70ox+lR48e2X///TN79uxyt0YTddttt2X99dfPLrvskv322y+dO3fOaaedVu62IEkyderU/OQnP8laa62VHXfcMb///e/L3RJN2PTp03PUUUdljTXWyMEHH5x+/fpls802y7hx48rdGk3U2LFjM3DgwGy00UY5/PDD07Nnz+y9995+VGClJwhhsV5++eUMHTo0Tz/9dLlbYSXw+uuv58QTT8yHH36Y5557LuPHj8/DDz+cX/ziF+VujSZqypQpueeeezJ+/Pi8/PLL+dvf/pZf/vKXGTVqVLlbg7z99tvp0aNHxo4dm7XWWqvc7dDETZ48OZtuumnef//9PP/885k4cWK6dOmSfffdt9yt0URNmDAhv/rVr/Luu+/m2WefzRtvvJHHHnss//3f/13u1mCZalHuBljxHXjggUn++Y8N+Lr22WefBt+7d++e3XffPWPGjClTRzR1Rx55ZIPv22yzTVq0aJHJkyeXqSP4/zbffPNsvvnm5W6DlUTfvn3Tt2/f0vdWrVrl0EMPzT777JPa2tpUVVWVsTuaot13373B9zXWWCN9+vQxh7LSE4QAZVVfX59nnnkma6+9drlboQn7+OOPM3bs2NTW1ubKK6/MRhttlO9///vlbgtgmXvqqafSqVMnIQhfWX19fR555JHU1dXlwQcfzGuvvZaLLrqo3G3BMiUIAcrq/PPPzwsvvJArrrii3K3QhI0fPz6nnHJKPvjgg0ydOjW/+c1v0q5du3K3BbBMPfHEE7nwwgtz3nnnlbsVmrD58+fnlFNOyaxZs/Laa6/l6KOPznrrrVfutmCZEoQAZfPnP/85P/vZz3L11Vdn4403Lnc7NGH9+/cv3V41evToDBo0KG3bts0PfvCDMncGsGy8/PLL2W233XLIIYfk2GOPLXc7NGGrrLJKaQ6dPHlytt1228yePdvDnVmpeVgqUBbXXnttDjvssPzxj3/MfvvtV+52WIlsv/32+c53vpO77rqr3K0ALBOvvPJKdthhh+y1117+scpS1a1bt/zwhz80h7LSE4QAy91f/vKX/OhHP8qVV16ZYcOGlbsdmrCFCxdm7ty5Dcbmz5+fd955J9/4xjfK1BXAsvPqq69mhx12yB577JHLL788FRUV5W6JJmzWrFmLjL3xxhvmUFZ6bo1hsaZNm5bx48dn+vTpSf75vvFWrVqlZ8+e6dmzZ5m7o6m57bbbcuCBB+bII4/MN7/5zdKlmJWVld6sQKPV1dVl8803z+GHH54NNtggH3/8ca644orMnj07//7v/17u9iDz58/PE088kSSZPXt2Jk2alDFjxqSqqirf/va3y9wdTc3bb7+dHXbYIb169cpBBx2URx55pLRs0003TevWrcvYHU3R/vvvn29/+9vp379/kuTOO+/MiBEjctNNN5W5M1i2Kurr6+vL3QQrtnvvvTdnnnnmIuMHH3xwjjjiiDJ0RFP2P//zP587ua6++uq5+eaby9ARTd27776biy++OM8991zatm2bTTbZJEcffbRfs1ghTJ8+PXvsscci4xtuuGEuv/zyMnREU/bwww9n+PDhn7vsuuuu8wMVjTZnzpxcfvnl+fvf/55PPvkk66yzTo488kgPS2WlJwgBAAAACsMzQgAAAIDCEIQAAAAAhSEIAQAAAApDEAIAAAAUhiAEAAAAKAxBCAAAAFAYghAAAACgMAQhwHLz7rvvZsSIEeVuAwCanIULF+b666/PBx98UO5WAJq8ivr6+vpyNwE0DR988EFGjRqVJKmoqEjHjh3Tt2/frL766ku0/u2335599tknc+fOXZZtAsAKacSIEVmwYEGSpH379unTp0/WWWedJVp37ty5ad26dR5++OFsvfXWy7JNgJVei3I3ADQd48aNy/7775/BgwenXbt2mThxYp555pmcffbZ+elPf1ru9gBghXbggQemb9++WWuttVJTU5O///3vGTx4cK6//vo0b9683O0BFIYgBGi0Cy+8MGuvvXaS5Pzzz89JJ52UIUOGpHfv3kmSKVOm5Omnn85qq62W73znO2nZsuXnbuejjz7KPffckySprKzMOuusk29961uL1NXU1OSpp55KfX19Nttss6y22mpLtAwAVjSHHXZYjj766CTJs88+m8033zxXXXVVDjvssCTJ7Nmz8/jjj2fevHnZYostsuqqq37htm6++ebMmzcvzZs3T48ePbLJJpuksrKyQc2CBQvyxBNPZPr06enbt2969eq1RMsAVmaCEOBrGTJkSE488cS89NJL6d27d04//fScd9552WyzzZIkc+bMye23357OnTsvsu5HH32UW2+9tVT3+OOPZ7PNNsutt95aCk9Gjx6dvfbaKxtuuGGqqqoybty4nH/++dlrr72+dBkArOj69euXnj175vnnn0+S3HHHHTnooIOyxhprpFu3bnnjjTfyhz/8Idttt93nrn/HHXdk1qxZWbBgQcaOHZsFCxbkzjvvLP1YMXXq1NK6ffr0ySuvvJK99947v/71r790GcDKThACfC2vv/56kqRr16658cYbc/bZZ+fBBx8s3b/8wgsvZM6cOZ+7bu/evXP99deXvs+YMSObbbZZrrzyyhx11FFJkl//+tf50Y9+lPPPPz9JMnPmzDz22GOLXQYAK7qamppMnTo1Xbt2zaRJkzJ06ND8/Oc/z89//vMkyfTp0/PKK6984fp/+MMfSv9dX1+fQw45JMOHDy89mPzPf/5zWrVqlWeeeSbNmjVLfX19Ro4cudhlACs7QQjQaHfccUc6d+6cyZMn5/zzz893v/vdbLbZZjnjjDMyZMiQBg9x22ijjb50WwsXLswzzzyTSZMmpa6uLj179syTTz5ZCkJat26dt956KzNnzky7du3Srl277LTTTotdBgAroqeffjrXX399ZsyYkSuvvDJt27bNoYcemuuvvz5t27bNKaecUqrt0KHDYh+MOn78+LzxxhuZMWNGOnTokNGjR5eWtW7dOh999FHeeeedfPOb30xFRUWGDBmy2GUAKztBCNBo9913X9q3b58OHTrkV7/6VQ444IA0a9Ys77zzTvbYY48l3s6ECROy884755NPPskGG2yQ9u3b55133kmrVq1KNb/+9a9z+OGHZ/XVV88WW2yRwYMH58c//nHatGnzpcsAYEX0wgsvlAL8vfbaKz/60Y/SsWPHvPPOO+ndu/cSPzR1/vz52XvvvTNmzJhsvvnmWXXVVTNlypRMnTq1VHPooYfmySefzIYbbpg+ffpkp512yr/927+lV69eX7oMYGUnCAEa7V8flvqvVl111Xz44YdLvJ1f/vKX6dOnT+64447S2AEHHJCPP/649H2dddbJ3//+93zwwQd54IEH8qtf/Sr33HNP7r333i9dBgAron99WOq/auwcOmLEiDz55JOZMGFC6YGqV111VZ588slSTevWrXPVVVfl97//fR577LH8/ve/T79+/fLGG29ktdVW+9JlACuzZuVuAFh57LzzzrnlllsyY8aM0ti8efNSW1v7ufVTpkzJuuuuW/peW1ub++67r0HN5MmTkyQdO3bMvvvum1NOOSWPP/74YpcBQFOy8847580338wjjzzSYHzatGmfWz9lypR07ty5wVtlbrrppgY1n86TrVu3zg477JBLL70006dPz+uvv/6lywBWdq4IAZaan/zkJ7n11lvTv3//HHHEEUmS6667LldffXWqqqoWqR8yZEj+4z/+I506dUpVVVUuv/zyzJs3r0HNYYcdlg4dOmSbbbbJggUL8rvf/S5Dhw5d7DIAaEoGDBiQY445JoMHD85xxx2Xbt265Y477shuu+2WI488cpH6733vexk+fHh+/OMfZ7PNNsvtt9+eMWPGNKj585//nDvvvDO77bZbOnXqlBtvvDHrrrtuvv3tb+fCCy/8wmUAK7uK+vr6+nI3ATQN48ePz+mnn57f/va3n/s63CSZO3du/vSnP+WJJ55Ip06dcuCBB+Zb3/pWkuS5557L//zP/+Tqq68u1Y8YMSL33ntvmjVrlp133jkzZ87M+++/n//8z/9MkixYsCAjRozIQw89lIqKimy99dYZOnRoWrRo8aXLAGBFc9BBB+Wggw7KwIEDv7Dm9ttvz5133pkkGTx4cAYPHpzkn88EOfDAA3PGGWeUrqZ85plncvXVV6empib9+vVLv379csUVVzSYZx955JGMHDky06dPz4YbbphDDjmkdOvLly0DWJkJQgAAAIDC8IwQAAAAoDAEIQAAAEBhCEIAAACAwhCEAAAAAIUhCAEAAAAKQxACAAAAFIYgBAAAACgMQQgAAABQGIIQAAAAoDAEIQAAAEBhCEIAAACAwhCEAAAAAIXx/wApId2NDAP+dQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1100x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "Survived  Sex   \n",
       "0         male      468\n",
       "          female     81\n",
       "1         female    233\n",
       "          male      109\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sns.catplot(data=df, x='Pclass', hue='Sex', col='Survived', kind='count')\n",
    "plt.show()\n",
    "\n",
    "count = df.groupby('Pclass')['Sex'].value_counts()\n",
    "pd.DataFrame(count)\n",
    "not_survive_count = df.groupby('Survived')['Sex'].value_counts()\n",
    "not_survive_count"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "56c3c0ef-83fc-43a7-8129-bb5045d5d31c",
   "metadata": {},
   "source": [
    "## Women survival %tage rate is more than men \n",
    "## Most of the people's age is between 20-40\n",
    "## 1st class passengers saved more than 3rd class \n",
    "## Survival Rate of 3rd class is 24% (75% of 3rd class passenger are not able to survived)\n",
    "## Total count of men Who not Survive is 468/577 And count of  Female 81/314"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ed3fcb42-4b30-48b2-9093-faec1d95996e",
   "metadata": {},
   "source": [
    "# Handling Null Value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "06f03e0e-ff2a-40bf-83a3-2b912b53a0e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "2a20d501-35e3-4ca4-a158-d285e5bd8569",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age             86\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             1\n",
       "Cabin          327\n",
       "Embarked         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "34a665a0-6c29-476f-a7d8-e62d40a7975f",
   "metadata": {},
   "outputs": [],
   "source": [
    "df= df.drop(columns = ['Cabin'])\n",
    "train = df\n",
    "test = df2.drop(columns = ['Cabin'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "bf44ed80-15b2-4272-8ee1-965479739119",
   "metadata": {},
   "outputs": [],
   "source": [
    "test['Fare'] = test['Fare'].fillna(test['Fare'].median())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "f0586a5a-9b06-4143-80fe-6a65ed4bed77",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId     0\n",
       "Pclass          0\n",
       "Name            0\n",
       "Sex             0\n",
       "Age            86\n",
       "SibSp           0\n",
       "Parch           0\n",
       "Ticket          0\n",
       "Fare            0\n",
       "Embarked        0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "c2dcf802-7c98-4ad6-8745-9ab40cec11ab",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "5021f7fb-692a-4bca-8414-7143aa2e471d",
   "metadata": {},
   "outputs": [],
   "source": [
    "train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "58ae7eff-7ca3-4ffc-9e5e-1ccbbd62b735",
   "metadata": {},
   "outputs": [],
   "source": [
    "train['Age'] = train['Age'].fillna(train['Age'].median())\n",
    "test['Age']= test['Age'].fillna(test['Age'].median())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "3d920fec-86ef-4488-9218-707478f06b32",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(PassengerId    0\n",
       " Survived       0\n",
       " Pclass         0\n",
       " Name           0\n",
       " Sex            0\n",
       " Age            0\n",
       " SibSp          0\n",
       " Parch          0\n",
       " Ticket         0\n",
       " Fare           0\n",
       " Embarked       0\n",
       " dtype: int64,\n",
       " PassengerId    0\n",
       " Pclass         0\n",
       " Name           0\n",
       " Sex            0\n",
       " Age            0\n",
       " SibSp          0\n",
       " Parch          0\n",
       " Ticket         0\n",
       " Fare           0\n",
       " Embarked       0\n",
       " dtype: int64)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.isnull().sum(), test.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "d557f726-d4cd-4316-926f-9dd828a3e275",
   "metadata": {},
   "outputs": [],
   "source": [
    "train = train.drop(columns = ['Name', 'Ticket','PassengerId'])\n",
    "test = test.drop(columns = ['Name', 'Ticket','PassengerId'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "e12ed42d-af86-44cc-b12f-c16d15655db7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<ArrowStringArray>\n",
       "['Q', 'S', 'C']\n",
       "Length: 3, dtype: str"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test['Embarked'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "267ce594-6b22-47a6-92d5-163d6d690a41",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>28.0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Survived  Pclass     Sex   Age  SibSp  Parch     Fare Embarked\n",
       "0           0       3    male  22.0      1      0   7.2500        S\n",
       "1           1       1  female  38.0      1      0  71.2833        C\n",
       "2           1       3  female  26.0      0      0   7.9250        S\n",
       "3           1       1  female  35.0      1      0  53.1000        S\n",
       "4           0       3    male  35.0      0      0   8.0500        S\n",
       "..        ...     ...     ...   ...    ...    ...      ...      ...\n",
       "886         0       2    male  27.0      0      0  13.0000        S\n",
       "887         1       1  female  19.0      0      0  30.0000        S\n",
       "888         0       3  female  28.0      1      2  23.4500        S\n",
       "889         1       1    male  26.0      0      0  30.0000        C\n",
       "890         0       3    male  32.0      0      0   7.7500        Q\n",
       "\n",
       "[891 rows x 8 columns]"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "612f897e-875c-4932-80ce-914603af88bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "train['Sex'] = train['Sex'].map({'male': 0 , 'female': 1})\n",
    "test['Sex'] = test['Sex'].map({'male': 0 , 'female': 1})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "20f4ee85-6b06-4341-b751-e66cb095ca8a",
   "metadata": {},
   "outputs": [],
   "source": [
    "train = pd.get_dummies(train, columns = ['Embarked'], drop_first = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "98f2e901-a8eb-43ae-b3b8-59ceca0aa182",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked_Q</th>\n",
       "      <th>Embarked_S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>39.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>108.9000</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>415</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>38.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>416</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>22.3583</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>418 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pclass  Sex   Age  SibSp  Parch      Fare  Embarked_Q  Embarked_S\n",
       "0         3    0  34.5      0      0    7.8292        True       False\n",
       "1         3    1  47.0      1      0    7.0000       False        True\n",
       "2         2    0  62.0      0      0    9.6875        True       False\n",
       "3         3    0  27.0      0      0    8.6625       False        True\n",
       "4         3    1  22.0      1      1   12.2875       False        True\n",
       "..      ...  ...   ...    ...    ...       ...         ...         ...\n",
       "413       3    0  27.0      0      0    8.0500       False        True\n",
       "414       1    1  39.0      0      0  108.9000       False       False\n",
       "415       3    0  38.5      0      0    7.2500       False        True\n",
       "416       3    0  27.0      0      0    8.0500       False        True\n",
       "417       3    0  27.0      1      1   22.3583       False       False\n",
       "\n",
       "[418 rows x 8 columns]"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test = pd.get_dummies (test, columns = ['Embarked'], drop_first = True)\n",
    "test"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d400ad2f-d64f-413d-ae53-eb12886b6c94",
   "metadata": {},
   "source": [
    "# Train Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "a4d46aa1-faa1-4f91-af49-33992ce3d5f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = train.drop(columns=['Survived'])\n",
    "y = train['Survived']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "1f192e69-905e-426d-a8e0-1a903dc35720",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "1fdcbc93-5150-4fcc-b6c0-220524c2eacd",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "3ddde1d5-318d-4fdb-95d0-b34582928cae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked_Q</th>\n",
       "      <th>Embarked_S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>709</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>28.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>15.2458</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>439</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>31.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>10.5000</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>840</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>720</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>33.0000</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>11.2417</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>433</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>17.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.1250</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>773</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2250</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>31.3875</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>17.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>10.5000</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>16.7000</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>179 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pclass  Sex   Age  SibSp  Parch     Fare  Embarked_Q  Embarked_S\n",
       "709       3    0  28.0      1      1  15.2458       False       False\n",
       "439       2    0  31.0      0      0  10.5000       False        True\n",
       "840       3    0  20.0      0      0   7.9250       False        True\n",
       "720       2    1   6.0      0      1  33.0000       False        True\n",
       "39        3    1  14.0      1      0  11.2417       False       False\n",
       "..      ...  ...   ...    ...    ...      ...         ...         ...\n",
       "433       3    0  17.0      0      0   7.1250       False        True\n",
       "773       3    0  28.0      0      0   7.2250       False       False\n",
       "25        3    1  38.0      1      5  31.3875       False        True\n",
       "84        2    1  17.0      0      0  10.5000       False        True\n",
       "10        3    1   4.0      1      1  16.7000       False        True\n",
       "\n",
       "[179 rows x 8 columns]"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "bfbf3bd8-0419-41df-ba0c-bdab572f1735",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score,recall_score, classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "b40e3144-be55-4767-a7bb-559c1c5a6e97",
   "metadata": {},
   "outputs": [],
   "source": [
    "lgr = LogisticRegression(max_iter = 1200)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "7e14beab-2f17-4e38-8e3d-2562f2c52259",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_lgr = lgr.fit(X_train,y_train)\n",
    "y_pred = model_lgr.predict(X_test)\n",
    "accuracy=accuracy_score(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "216b6301-7860-4b86-aa3b-aecb2b57d5f3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[90, 15],\n",
       "       [19, 55]])"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(y_test,y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "7f67f1f0-dd2a-4f01-aff4-4faa50bf3d3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "f1 = f1_score(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "828ff0e0-8186-47cc-8b17-be4c9c48f3fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "precision = precision_score(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "43726c8c-4a8a-47e2-9201-2878d1a59341",
   "metadata": {},
   "outputs": [],
   "source": [
    "recall = recall_score(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "8c2f556c-cffe-45ac-83a8-f50c6847def4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " accuracy: 0.81\n",
      " f1-accuracy: 0.76\n",
      " precision: 0.79\n",
      "recall accuracy : 0.74\n"
     ]
    }
   ],
   "source": [
    "print(' accuracy:',np.round(accuracy,2))\n",
    "print(' f1-accuracy:',np.round(f1, 2))\n",
    "print(' precision:',np.round(precision,2))\n",
    "print('recall accuracy :',np.round(recall,2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "c763a3f7-d291-47be-a4bf-1da5ccfbf13d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.83      0.86      0.84       105\n",
      "           1       0.79      0.74      0.76        74\n",
      "\n",
      "    accuracy                           0.81       179\n",
      "   macro avg       0.81      0.80      0.80       179\n",
      "weighted avg       0.81      0.81      0.81       179\n",
      "\n"
     ]
    }
   ],
   "source": [
    "report=classification_report(y_test,y_pred)\n",
    "print(report)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "2536e43d-2b9c-4eb2-b3fe-5a4368bb6049",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "specificity: 0.86\n"
     ]
    }
   ],
   "source": [
    "tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()\n",
    "specificity = tn/(tn + fp) \n",
    "print('specificity:',np.round(specificity,2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "b0191ac1-3b38-4bec-bff9-c60c2dbc27a7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(np.int64(90), np.int64(19), np.int64(55), np.int64(15))"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tn,fn,tp,fp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "54e5b70f-2893-46c2-a6cb-7c2af1e8bb93",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.09805301, 0.91070391, 0.63323281, 0.87739003, 0.08956024,\n",
       "       0.13978512, 0.28680663, 0.0935585 , 0.57632796, 0.88205811,\n",
       "       0.69806238, 0.8163033 , 0.13469219, 0.0385166 , 0.7136273 ,\n",
       "       0.64973646, 0.0945623 , 0.23977945, 0.53081457, 0.70730468,\n",
       "       0.2084109 , 0.20792836, 0.76308114, 0.46066378, 0.53384488,\n",
       "       0.35516604, 0.15345366, 0.40158797, 0.68385102, 0.10858761,\n",
       "       0.46365032, 0.94386477, 0.68377899, 0.08924076, 0.51654099,\n",
       "       0.30171321, 0.15345507, 0.13116603, 0.55612408, 0.73600632,\n",
       "       0.45662875, 0.76705015, 0.1536785 , 0.8968046 , 0.74017539,\n",
       "       0.10862611, 0.10962088, 0.68377899, 0.09432815, 0.6272956 ,\n",
       "       0.06466887, 0.13109258, 0.88077183, 0.75830928, 0.28287942,\n",
       "       0.46066378, 0.83806818, 0.15147835, 0.84106186, 0.03995834,\n",
       "       0.1788459 , 0.90389099, 0.29948008, 0.08146803, 0.55514545,\n",
       "       0.10999217, 0.80205138, 0.13833133, 0.35952793, 0.06701415,\n",
       "       0.2170884 , 0.32256305, 0.3135193 , 0.12745658, 0.10887276,\n",
       "       0.11773054, 0.10858761, 0.10862611, 0.37814247, 0.60717771,\n",
       "       0.12799173, 0.10605295, 0.68379989, 0.46810375, 0.85399738,\n",
       "       0.36941614, 0.09198458, 0.10862611, 0.88783981, 0.12105549,\n",
       "       0.10569899, 0.13463335, 0.2813301 , 0.07425192, 0.0449868 ,\n",
       "       0.10862611, 0.2542224 , 0.58863309, 0.76321512, 0.16803228,\n",
       "       0.61889149, 0.10858761, 0.5140494 , 0.09481012, 0.04871014,\n",
       "       0.10858761, 0.66782763, 0.10855746, 0.08232273, 0.62662885,\n",
       "       0.33260517, 0.73464132, 0.12771857, 0.60799162, 0.77513291,\n",
       "       0.1311293 , 0.04232282, 0.18853518, 0.69053283, 0.48543429,\n",
       "       0.20179382, 0.10862611, 0.24158033, 0.78558743, 0.27819788,\n",
       "       0.18189034, 0.13956564, 0.1208065 , 0.62658993, 0.0673745 ,\n",
       "       0.13481838, 0.13439193, 0.40730238, 0.75830928, 0.2569075 ,\n",
       "       0.35516179, 0.92186354, 0.33551578, 0.14998672, 0.61694673,\n",
       "       0.66528593, 0.66106394, 0.58225437, 0.17564634, 0.29905367,\n",
       "       0.22775524, 0.11155918, 0.58101796, 0.16847207, 0.1704817 ,\n",
       "       0.13484346, 0.91686453, 0.04991589, 0.06376108, 0.1084421 ,\n",
       "       0.37673207, 0.75736531, 0.10284165, 0.10877913, 0.01068734,\n",
       "       0.06406499, 0.7457665 , 0.11462234, 0.14594956, 0.07670008,\n",
       "       0.15349437, 0.91487545, 0.32926115, 0.45453562, 0.12132652,\n",
       "       0.23641084, 0.0894516 , 0.71412802, 0.1311293 , 0.34807713,\n",
       "       0.09946834, 0.04499859, 0.89482933, 0.22880365, 0.08546028,\n",
       "       0.1258837 , 0.32095433, 0.0540516 , 0.27683374, 0.73875063,\n",
       "       0.46996609, 0.62139563, 0.33162106, 0.07110792, 0.08706507,\n",
       "       0.78815287, 0.29348141, 0.61405527, 0.31881293, 0.91068767,\n",
       "       0.90025622, 0.13956564, 0.0665892 , 0.68377899, 0.82615037,\n",
       "       0.10898867, 0.01068734, 0.09175192, 0.09593967, 0.14198459,\n",
       "       0.76474909, 0.07566073, 0.16565205, 0.75737323, 0.46575375,\n",
       "       0.12078139, 0.77602871, 0.12748892, 0.22880365, 0.10768548,\n",
       "       0.93368393, 0.62609866, 0.13683644, 0.94345148, 0.22766812,\n",
       "       0.14960356, 0.24540076, 0.05686474, 0.10858761, 0.3500578 ,\n",
       "       0.12809249, 0.29214649, 0.13268163, 0.2998651 , 0.38580369,\n",
       "       0.88557013, 0.10563198, 0.10899571, 0.46262638, 0.26154455,\n",
       "       0.61868115, 0.12947752, 0.86601788, 0.29214649, 0.21269895,\n",
       "       0.64686089, 0.62139563, 0.23309256, 0.12745307, 0.1456722 ,\n",
       "       0.30760662, 0.64021995, 0.79353754, 0.31157089, 0.09871942,\n",
       "       0.10842652, 0.51427912, 0.22776488, 0.08010027, 0.47571936,\n",
       "       0.65843895, 0.94999457, 0.92433409, 0.97910328, 0.66565672,\n",
       "       0.13956564, 0.0642432 , 0.23472184, 0.35061183, 0.68377899,\n",
       "       0.19700892, 0.0498797 , 0.09034101, 0.84693039, 0.92247563,\n",
       "       0.45778211, 0.11569694, 0.72055143, 0.46087929, 0.68377899,\n",
       "       0.76411745, 0.4910902 , 0.23372223, 0.08225209, 0.47472187,\n",
       "       0.0496934 , 0.10857723, 0.15007988, 0.13829804, 0.45458357,\n",
       "       0.13504915, 0.10318715, 0.12767427, 0.1704817 , 0.72207196,\n",
       "       0.93120822, 0.95048088, 0.26898919, 0.64786913, 0.12101319,\n",
       "       0.55514545, 0.17220519, 0.95326365, 0.45746209, 0.93068377,\n",
       "       0.68377899, 0.08546283, 0.13584279, 0.84829029, 0.10862611,\n",
       "       0.61261464, 0.95373608, 0.95528721, 0.25290053, 0.94404926,\n",
       "       0.95593025, 0.94858985, 0.75531831, 0.10858761, 0.12106507,\n",
       "       0.63319041, 0.78522963, 0.12490521, 0.92115604, 0.9101974 ,\n",
       "       0.12748892, 0.11158474, 0.84024663, 0.77856037, 0.01068734,\n",
       "       0.94506417, 0.04232394, 0.76699987, 0.50547224, 0.95885772,\n",
       "       0.55471972, 0.32934888, 0.4335277 , 0.09083685, 0.91604748,\n",
       "       0.10858761, 0.40040653, 0.93638576, 0.06754889, 0.33675617,\n",
       "       0.32549377, 0.88475683, 0.23977945, 0.2569075 , 0.19803074,\n",
       "       0.82615037, 0.74441966, 0.55238206, 0.15142402, 0.07367231,\n",
       "       0.12467903, 0.46034346, 0.1526767 , 0.09248803, 0.15345366,\n",
       "       0.10898867, 0.92812525, 0.75588649, 0.68385102, 0.68385102,\n",
       "       0.04114451, 0.26056913, 0.5677856 , 0.0893502 , 0.10962088,\n",
       "       0.10265147, 0.84015766, 0.70730692, 0.68377899, 0.95440002,\n",
       "       0.5222473 , 0.10923172, 0.13829804, 0.66447918, 0.57164386,\n",
       "       0.93439723, 0.66077499, 0.62478451, 0.18674643, 0.13821356,\n",
       "       0.94777092, 0.81969471, 0.09730327, 0.87708456, 0.10858761,\n",
       "       0.3336033 , 0.05349488, 0.76699987, 0.1395592 , 0.89746637,\n",
       "       0.33263129, 0.13109135, 0.0631756 , 0.94732317, 0.60201219,\n",
       "       0.12764556, 0.5969936 , 0.15828099, 0.26749657, 0.80772641,\n",
       "       0.0800463 , 0.11469432, 0.6006761 , 0.08467642, 0.6751523 ,\n",
       "       0.16623755, 0.05682327, 0.3147667 , 0.13108524, 0.38580369,\n",
       "       0.10858761, 0.13928975, 0.9176629 , 0.23372223, 0.06948079,\n",
       "       0.61898527, 0.71078165, 0.82143256, 0.22880365, 0.70295211,\n",
       "       0.1536785 , 0.16731443, 0.10565634, 0.52445055, 0.10235952,\n",
       "       0.10842652, 0.7638722 , 0.85133152, 0.13956564, 0.09733159,\n",
       "       0.45493523, 0.55238206, 0.67824084, 0.14545606, 0.25467061,\n",
       "       0.92869903, 0.48995514, 0.65878901, 0.19776999, 0.22233333,\n",
       "       0.63332858, 0.13512857, 0.09034101, 0.80786654, 0.10864171,\n",
       "       0.61776576, 0.85861755, 0.40991451, 0.71455319, 0.28805672,\n",
       "       0.13326582, 0.08550283, 0.54000638, 0.36391186, 0.10862611,\n",
       "       0.14974131, 0.2120254 , 0.89834015, 0.68063094, 0.13956564,\n",
       "       0.31160059, 0.09208644, 0.32492326, 0.14606711, 0.10862611,\n",
       "       0.08215811, 0.23372223, 0.26165083, 0.1395579 , 0.74031953,\n",
       "       0.10842652, 0.08247221, 0.69200161, 0.87978923, 0.66226898,\n",
       "       0.47125095, 0.16623755, 0.08063591, 0.12756662, 0.76559469,\n",
       "       0.0423729 , 0.23372223, 0.058528  , 0.35856167, 0.54511305,\n",
       "       0.38580369, 0.88726413, 0.3337529 , 0.10284165, 0.12931321,\n",
       "       0.08550283, 0.13093112, 0.26960542, 0.26155054, 0.13116603,\n",
       "       0.15589236, 0.86421512, 0.11039892, 0.90984036, 0.12098576,\n",
       "       0.14594956, 0.72816996, 0.68371163, 0.55326074, 0.94936133,\n",
       "       0.60851201, 0.75038801, 0.45493523, 0.11229338, 0.12800234,\n",
       "       0.13593195, 0.10862611, 0.39503759, 0.8584043 , 0.12090352,\n",
       "       0.32239487, 0.77663977, 0.14472192, 0.71692682, 0.09729666,\n",
       "       0.92558692, 0.12767427, 0.15345366, 0.90821649, 0.15345507,\n",
       "       0.09963039, 0.68063094, 0.57997636, 0.0800463 , 0.15391468,\n",
       "       0.8654668 , 0.15345507, 0.1449279 , 0.66935756, 0.60483042,\n",
       "       0.86952825, 0.33162106, 0.95042104, 0.11024706, 0.9445505 ,\n",
       "       0.88726821, 0.43230734, 0.41735688, 0.17676103, 0.36717481,\n",
       "       0.21694091, 0.80990317, 0.32028732, 0.06727037, 0.29224894,\n",
       "       0.63561933, 0.25166125, 0.13959017, 0.17884431, 0.66107838,\n",
       "       0.22776488, 0.87375897, 0.67627512, 0.85887662, 0.49222391,\n",
       "       0.13956564, 0.07781668, 0.24001456, 0.10862611, 0.61898527,\n",
       "       0.07362054, 0.13825066, 0.51401925, 0.15345507, 0.09728724,\n",
       "       0.0997001 , 0.75349767, 0.39509921, 0.68377899, 0.14960356,\n",
       "       0.14029208, 0.77775607, 0.86445364, 0.64686331, 0.09730327,\n",
       "       0.7627997 , 0.90768942, 0.12832692, 0.50217743, 0.1539526 ,\n",
       "       0.93315213, 0.15058319, 0.26336642, 0.12771857, 0.10862611,\n",
       "       0.08936594, 0.87123689, 0.06367219, 0.63532301, 0.1555889 ,\n",
       "       0.06227874, 0.81574474, 0.05907239, 0.15345366, 0.34493168,\n",
       "       0.70997197, 0.10858761, 0.46508738, 0.06950163, 0.5010749 ,\n",
       "       0.06748543, 0.10280497, 0.46506475, 0.82933352, 0.91446936,\n",
       "       0.34812473, 0.10837668, 0.62139563, 0.13956564, 0.08956024,\n",
       "       0.76510994, 0.06441746, 0.56745857, 0.82318675, 0.24989996,\n",
       "       0.12409268, 0.30201437, 0.13648655, 0.1311085 , 0.13354982,\n",
       "       0.23586237, 0.14728035, 0.94025545, 0.11465395, 0.13956047,\n",
       "       0.14644169, 0.05672718, 0.52654249, 0.43801735, 0.50363876,\n",
       "       0.80786654, 0.09730327, 0.1658618 , 0.40827822, 0.0847264 ,\n",
       "       0.13463335, 0.95440002, 0.55693188, 0.12132652, 0.74031953,\n",
       "       0.36360734, 0.13825066, 0.35089354, 0.10850133, 0.65405881,\n",
       "       0.10858761, 0.84022536, 0.13127871, 0.68382314, 0.74546379,\n",
       "       0.18741349, 0.10858761, 0.56587994, 0.26876162, 0.33362796,\n",
       "       0.23702356, 0.11155998, 0.31766269, 0.08698312, 0.10376098,\n",
       "       0.15294801, 0.2569075 , 0.10855746, 0.07150648, 0.89837252,\n",
       "       0.67532629, 0.37692765, 0.07978145, 0.22344995, 0.23372223,\n",
       "       0.14189823, 0.11943724, 0.68907798, 0.30631233, 0.75367672,\n",
       "       0.683995  , 0.59349986, 0.13504571, 0.03658226, 0.07801572,\n",
       "       0.24955835, 0.05286077, 0.13895106, 0.14190476, 0.95990351,\n",
       "       0.37996043, 0.82120553, 0.12132652, 0.16575979, 0.23870628,\n",
       "       0.13160101, 0.06950163, 0.68376967, 0.35199057, 0.07349433,\n",
       "       0.96567816, 0.40237222, 0.75005087, 0.15095389, 0.08787442,\n",
       "       0.18894034, 0.7145086 , 0.35211517, 0.94859494, 0.10999217,\n",
       "       0.95212621, 0.45493523, 0.26449749, 0.10604887, 0.13145373,\n",
       "       0.13817517, 0.95350735, 0.81158695, 0.14198285, 0.09461573,\n",
       "       0.88623333, 0.11238795, 0.20792836, 0.13859889, 0.40675573,\n",
       "       0.13487637, 0.62446442, 0.68377202, 0.21010648, 0.56979238,\n",
       "       0.94558952, 0.23906036, 0.23372223, 0.26876162, 0.26876162,\n",
       "       0.10915597, 0.33657816, 0.77850053, 0.10858761, 0.10858761,\n",
       "       0.45714211, 0.3574734 , 0.94392498, 0.09471167, 0.10002389,\n",
       "       0.14743637, 0.10812945, 0.7981895 , 0.46688991, 0.1289037 ,\n",
       "       0.85723747, 0.1782293 , 0.09499747, 0.12430545, 0.60983651,\n",
       "       0.32792153, 0.10856265, 0.29905367, 0.09208644, 0.91765934,\n",
       "       0.11024706, 0.07551044, 0.18800944, 0.86918679, 0.14951337,\n",
       "       0.82383162, 0.562697  , 0.66701302, 0.111817  , 0.09740241,\n",
       "       0.12145389, 0.06196285, 0.63239582, 0.15345366, 0.51167713,\n",
       "       0.14189038, 0.13956564, 0.76858454, 0.13956177, 0.91042927,\n",
       "       0.7926976 , 0.92612999, 0.44956013, 0.07063902, 0.11756993,\n",
       "       0.11762345, 0.68778046, 0.0799719 , 0.14700756, 0.45103777,\n",
       "       0.13956564, 0.32011829, 0.1258837 , 0.55703918, 0.11779639,\n",
       "       0.1838568 , 0.85386634, 0.59750786, 0.14567355, 0.51545314,\n",
       "       0.20792836, 0.72608544, 0.51713422, 0.27514671, 0.11134957,\n",
       "       0.09998908, 0.35760921, 0.68793504, 0.1838568 , 0.88382208,\n",
       "       0.11465178, 0.08318187, 0.20189382, 0.45495774, 0.10143878,\n",
       "       0.43801735, 0.65427757, 0.23356317, 0.07123304, 0.06874461,\n",
       "       0.80906892, 0.11178088, 0.36466781, 0.60329488, 0.0745613 ,\n",
       "       0.1393181 , 0.12132652, 0.47931484, 0.13956564, 0.81860858,\n",
       "       0.7316485 , 0.32925937, 0.15345507, 0.12429378, 0.14206315,\n",
       "       0.90152195, 0.13134613, 0.10862611, 0.10887276, 0.55640548,\n",
       "       0.13465462, 0.31148259, 0.94041092, 0.12913106, 0.14594956,\n",
       "       0.07347678, 0.01068734, 0.12783919, 0.22963875, 0.93548471,\n",
       "       0.06241132, 0.02894669, 0.74660318, 0.93712006, 0.6647444 ,\n",
       "       0.66518977, 0.86320453, 0.29226029, 0.67076137, 0.15345507,\n",
       "       0.04400848, 0.22455881, 0.85764274, 0.1258837 , 0.26279155,\n",
       "       0.73260475, 0.82814346, 0.44749761, 0.10898867, 0.14599532,\n",
       "       0.11465395, 0.81628253, 0.40388736, 0.06394176, 0.8275102 ,\n",
       "       0.78246044, 0.13523281, 0.13825066, 0.10858761, 0.87972616,\n",
       "       0.81053307, 0.09464242, 0.6626606 , 0.2386063 , 0.11756993,\n",
       "       0.48739983, 0.24540076, 0.93664771, 0.50329521, 0.57164105,\n",
       "       0.12550711])"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_lgr.predict_proba(X)[:,1]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d2b0d933-92e1-47a3-b2a2-ddb9ce2f351f",
   "metadata": {},
   "source": [
    "# ROC AND AUC "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "fafc0483-ba8b-40e6-872b-b9999520a5bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import roc_curve, auc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "b427ec25-be88-45c0-a4fb-c63a506f33a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model_lgr.predict(X_test)\n",
    "predict_proba = model_lgr.predict_proba(X_test)[:,1]\n",
    "\n",
    "fpr,tpr,threshold = roc_curve(y_test, y_pred)\n",
    "roc_auc = auc(fpr, tpr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "f350a214-8fbc-4574-9426-d043a4233dad",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure()\n",
    "plt.plot(fpr, tpr,color = 'darkorange', lw = 2, label = 'roc_curve(area = %0.2f)' % roc_auc)\n",
    "plt.plot([0,1], [0,1], color = 'navy', lw = 2, linestyle = '--')\n",
    "plt.xlim([0.0,1.0])\n",
    "plt.ylim([0.0, 1.05])\n",
    "plt.legend(loc = 'lower right')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "a148844f-4f6e-410e-bc28-6090e4be95e9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Features</th>\n",
       "      <th>Coeficient</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Pclass</td>\n",
       "      <td>-0.938199</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Sex</td>\n",
       "      <td>2.590096</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Age</td>\n",
       "      <td>-0.030595</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>SibSp</td>\n",
       "      <td>-0.295702</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Parch</td>\n",
       "      <td>-0.108036</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Fare</td>\n",
       "      <td>0.002579</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Embarked_Q</td>\n",
       "      <td>-0.112490</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Embarked_S</td>\n",
       "      <td>-0.399213</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Features  Coeficient\n",
       "0      Pclass   -0.938199\n",
       "1         Sex    2.590096\n",
       "2         Age   -0.030595\n",
       "3       SibSp   -0.295702\n",
       "4       Parch   -0.108036\n",
       "5        Fare    0.002579\n",
       "6  Embarked_Q   -0.112490\n",
       "7  Embarked_S   -0.399213"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "coef_logr = pd.DataFrame({\n",
    "    'Features':X_train.columns,\n",
    "    'Coeficient':model_lgr.coef_[0]\n",
    "})\n",
    "coef_logr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "1af89cfd-bbf7-4f83-af88-d90f5e2f12a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.7877095  0.78651685 0.78651685 0.76966292 0.8258427 ] 0.7912497646098802\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "cv_scores = cross_val_score(lgr, X, y, cv=5, scoring='accuracy')\n",
    "print(cv_scores, cv_scores.mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "9941517c-62e5-4e66-b7a0-3ddf7a093c1c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['logistic_model.pkl']"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    "joblib.dump(model_lgr, 'logistic_model.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "fda924b5-a67f-452c-8910-7cd4b5a7930f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked_Q</th>\n",
       "      <th>Embarked_S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>331</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>45.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>28.5000</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>733</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>23.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>382</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>704</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.8542</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>813</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>31.2750</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>106</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>21.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.6500</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>270</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>31.0000</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>860</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>41.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>14.1083</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>120.0000</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>102</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>21.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>77.2875</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>712 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pclass  Sex   Age  SibSp  Parch      Fare  Embarked_Q  Embarked_S\n",
       "331       1    0  45.5      0      0   28.5000       False        True\n",
       "733       2    0  23.0      0      0   13.0000       False        True\n",
       "382       3    0  32.0      0      0    7.9250       False        True\n",
       "704       3    0  26.0      1      0    7.8542       False        True\n",
       "813       3    1   6.0      4      2   31.2750       False        True\n",
       "..      ...  ...   ...    ...    ...       ...         ...         ...\n",
       "106       3    1  21.0      0      0    7.6500       False        True\n",
       "270       1    0  28.0      0      0   31.0000       False        True\n",
       "860       3    0  41.0      2      0   14.1083       False        True\n",
       "435       1    1  14.0      1      2  120.0000       False        True\n",
       "102       1    0  21.0      0      1   77.2875       False        True\n",
       "\n",
       "[712 rows x 8 columns]"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2b3be853-565d-4c95-8166-3125cca06b64",
   "metadata": {},
   "source": [
    "1. What is the difference between precision and recall?\n",
    "\n",
    "   precision is the performance metric for classification model ,its predicted as positive how many were actually positive\n",
    "   ,its minimizes false positives,high precision means that when your model predicts positive, you can trust it.\n",
    "\n",
    "\n",
    "    formula:- True Positive/True Positive + False Positive\n",
    "    \n",
    "\n",
    "3. What is cross-validation, and why is it important in binary classification?\n",
    "\n",
    "\n",
    "    recall is the performance metrics for classification model, its also known as sensitivity, out of all the actual positive instances in the dataset, how many did the model manage to find?\n",
    "\n",
    "   formuls:- True Positive / True Positive + False Negative\n",
    "\n",
    "\n",
    "   \n",
    "\n",
    "   \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b207ee86-44e9-49f2-9e1f-1316d7cae7bf",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

# -*- coding: utf-8 -*-
"""log_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fi15DcVRxjGZZxihZPWfAHjvcnd1sMni
"""

import joblib
import pandas as pd
import streamlit as st

model = joblib.load('logistic_model.pkl')
st.title('Titanic survival prediction model')
st.write('Enter passenger details to predict survival')

# user inputs
Pclass = st.selectbox('Passenger Class', [1, 2, 3])
Sex = st.selectbox('Sex', ['male', 'female'])
Age = st.slider('Age', 0, 80, 25)
SibSp = st.number_input(
    'number of siblings/spouses:', min_value=0, max_value=10, value=0
)
Parch = st.number_input(
    'number of parents/children:', min_value=0, max_value=10, value=0
)
Fare = st.number_input('Fare paid:', min_value=0, max_value=1000, value=0)
Embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])

sex_encode = 1 if Sex == 'female' else 0
Embarked_encode_S = 1 if Embarked == 'S' else 0
Embarked_encode_Q = 1 if Embarked == 'Q' else 0

input_data = pd.DataFrame(
    {
        'Pclass': [Pclass],
        'Sex': [sex_encode],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Fare': [Fare],
        'Embarked_S': [Embarked_encode_S],
        'Embarked_Q': [Embarked_encode_Q],
    }
)

if st.button('Predict'):
  # FIXED: Using .values converts the dataframe into a raw array,
  # bypassing the strict column name checks that were throwing the error.
  prediction = model.predict(input_data.values)
  prediction_proba = model.predict_proba(input_data.values)

  survived_prob = prediction_proba[0][1]
  not_survived_prob = prediction_proba[0][0]

  if prediction[0] == 1:
    st.write(f'Survived with probability: {survived_prob:.2%}')
  else:
    st.write(f'Not Survived with probability: {not_survived_prob:.2%}')
       main
