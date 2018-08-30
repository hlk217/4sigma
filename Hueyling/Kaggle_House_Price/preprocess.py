class LabelCountEncoder(object):
	def __init__(self):
		self.count_dict = {}
		self.rev_count_dict = {}
	def fit(self, column):
		# We want to rank the key by its value and use the rank as the new value
		count = column.value_counts()
		self.count_dict = dict( list( zip (count.index, reversed(range(len(count)+1 ) ) ) ) ) 
		self.rev_count_dict = dict( list( zip ( reversed(range(len(count)+1 ) ) , count.index ) ) ) 
	def transform(self, column):
		# If a category only appears in the test set, we will assign the value to zero.
		missing = 0
		return column.map(lambda x: self.count_dict.get(x, missing))
	def fit_transform(self, column):
		self.fit(column)
		return self.transform(column)



import numpy as np
import pandas as pd
import math
import re
from scipy.stats import kurtosis, skew
from scipy.special import boxcox1p


def impute( inputDF, onehot = False, isBoxCox = True, skipFeature = ""):
	
	#input: pd.dataframe
	#one-hot or label encoding for the categorical field
	#return:
	#	Imputed pd.DataFrame
	#	label-encoded dictionary
	
	encodedDic = {}
	
	inputDF.Exterior1st = inputDF.Exterior1st.str.replace("Brk Cmn", "BrkComm")
	inputDF.Exterior2nd = inputDF.Exterior2nd.str.replace("Brk Cmn", "BrkComm")
	inputDF.Exterior1st = inputDF.Exterior1st.str.replace("CmentBd", "CemntBd")
	inputDF.Exterior2nd = inputDF.Exterior2nd.str.replace("CmentBd", "CemntBd")
	inputDF.Exterior1st = inputDF.Exterior1st.str.replace("Wd Shng", "WdShing")
	inputDF.Exterior2nd = inputDF.Exterior2nd.str.replace("Wd Shng", "WdShing")
	
	############################### purposelyEncodeData  ########################################
	## Start to purposely encode the information based on our best understanding.
	## Combine Exterior1st and Exterior2nd to Exterior
	## Add TotalFlrSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF
	## BsmtFinType1 and BsmtFinType2 to Bsmt -Replace each type to it's actually square feet BsmtFinSF1, BsmtFinSF2 -For Unf in type 1 and type2, replace it with the BsmtUnfSF
	## Combine BsmtFullBath, BsmtHalfBath to BsmtBath
	## Add all different PorchSF to TotalProchSF
	## Dummy MasVnrType to MasVnr and replace the value with MasVnrArea
	###############################################################################################
	
	preProcessCatField = ["MasVnrType", "Exterior1st", "Exterior2nd", "BsmtFinType1", "BsmtFinType2", "Condition1", "Condition2"]
	preProcessNumFiled = ["1stFlrSF", "2ndFlrSF", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath"]
	inputDF[preProcessCatField] = inputDF[preProcessCatField].fillna("Unknown")
	inputDF[preProcessNumFiled] = inputDF[preProcessNumFiled].fillna(0)
	
	# Exterior1st, Exterior2nd (Exterior covering on house)
	var1_dummy_columns = pd.get_dummies(inputDF['Exterior1st'], prefix= "Exterior")
	var2_dummy_columns = pd.get_dummies(inputDF['Exterior2nd'], prefix= "Exterior")
	var_dummy_columns = pd.concat([var1_dummy_columns,var2_dummy_columns], join='outer', sort=True).groupby(level=0).sum()
	var_dummy_columns = var_dummy_columns.replace(2, 1)

	inputDF = pd.concat([inputDF, var_dummy_columns], join='outer', sort=True, axis=1)
	inputDF = inputDF.drop( columns=['Exterior1st', 'Exterior2nd'] )
	
	#TotalBsmtFinSF = "BsmtFinSF1"+ "BsmtFinSF2" ( + "TotalBsmtSF" )
	inputDF["TotalBsmtSF"] = inputDF["TotalBsmtSF"].fillna(0)
	inputDF["TotalFlrSF"] = inputDF["1stFlrSF"] + inputDF["2ndFlrSF"] + inputDF["TotalBsmtSF"]

	# BsmtFinType1, BsmtFinType2, BsmtFinSF1 (Type 1 finished square feet), BsmtFinSF2 (Type 1 finished square feet), BsmtUnfSF: Unfinished square feet of basement area
	var1_dummy_columns = pd.get_dummies(inputDF['BsmtFinType1'], prefix= "Bsmt") 
	var1_dummy_columns = var1_dummy_columns.mul( inputDF['BsmtFinSF1'] , axis=0)
	tmp = var1_dummy_columns['Bsmt_Unf']
	tmp [ inputDF.loc[inputDF['BsmtFinType1'] == "Unf"].index ] = 1
	tmp = tmp.mul( inputDF['BsmtUnfSF'] , axis=0)
	var1_dummy_columns['Bsmt_Unf'] = tmp

	var2_dummy_columns = pd.get_dummies(inputDF['BsmtFinType2'], prefix= "Bsmt") 
	var2_dummy_columns = var2_dummy_columns.mul( inputDF['BsmtFinSF2'] , axis=0)
	tmp = var2_dummy_columns['Bsmt_Unf']
	tmp [ inputDF.loc[inputDF['BsmtFinType2'] == "Unf"].index ] = 1
	tmp = tmp.mul( inputDF['BsmtUnfSF'] , axis=0)
	var2_dummy_columns['Bsmt_Unf'] = tmp

	var_dummy_columns = pd.concat([var1_dummy_columns,var2_dummy_columns], join='outer', sort=True).groupby(level=0).sum()
	inputDF = pd.concat([inputDF, var_dummy_columns], join='outer', sort=True, axis=1)
	inputDF = inputDF.drop( columns=['BsmtFinType1', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'] )

	#BsmtFullBath, BsmtHalfBath  (number of type of bathroom in the basement)
	inputDF['BsmtBath'] = inputDF["BsmtFullBath"] + 0.5* inputDF["BsmtHalfBath"] 
	inputDF = inputDF.drop( columns=['BsmtFullBath', 'BsmtHalfBath'] )

	inputDF["TotalPorchSF"] = inputDF["OpenPorchSF"] + inputDF["EnclosedPorch"] + inputDF["3SsnPorch"] + inputDF["ScreenPorch"] 

	#MasVnrType, MasVnrArea
	var_dummy_columns = pd.get_dummies(inputDF['MasVnrType'], prefix= "MasVnr") 
	var_dummy_columns = var_dummy_columns.mul( inputDF['MasVnrArea'] , axis=0)
	inputDF = pd.concat([inputDF, var_dummy_columns], join='outer', sort=True, axis=1)
	inputDF = inputDF.drop( columns=['MasVnrType', 'MasVnrArea'] )
	
	#Condition1, Condition2: Proximity to various conditions
	var1_dummy_columns = pd.get_dummies(inputDF['Condition1'], prefix= "Cond")
	var2_dummy_columns = pd.get_dummies(inputDF['Condition2'], prefix= "Cond")
	var_dummy_columns = pd.concat([var1_dummy_columns,var2_dummy_columns], join='outer', sort=True).groupby(level=0).sum()
	var_dummy_columns = var_dummy_columns.replace(2, 1)
	inputDF = pd.concat([inputDF, var_dummy_columns], join='outer', sort=True, axis=1)
	inputDF = inputDF.drop( columns=['Condition1', 'Condition2'] )


	############################### Median Impute  #######################
	## Some NA existed in these numerical fields
#	impute_cols = ['GarageArea', 'GarageCars','GarageYrBlt','LotFrontage']
#	for i, c in enumerate ( impute_cols ):
#		if inputDF[c].isnull().any():
#			inputDF[c] = inputDF[c].fillna( inputDF[c].median() )


	# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
	# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
	inputDF["LotFrontage"] = inputDF.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

	# GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
	for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
		inputDF[col] = inputDF[col].fillna(0)

	## Some NA existed in these categorical fields.
	# Functional : data description says NA means typical
	inputDF["Functional"] = inputDF["Functional"].fillna("Typ")

	# Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
	inputDF['Electrical'] = inputDF['Electrical'].fillna(inputDF['Electrical'].mode()[0])

	# KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
	inputDF['KitchenQual'] = inputDF['KitchenQual'].fillna(inputDF['KitchenQual'].mode()[0])

	# SaleType : Fill in again with most frequent which is "WD"
	inputDF['SaleType'] = inputDF['SaleType'].fillna(inputDF['SaleType'].mode()[0])


	############################### Ordinal Features Label encoding  #######################

	ord_cols = ['ExterQual', 'ExterCond','BsmtCond','HeatingQC', 'KitchenQual', 
	           'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC', 'BsmtQual']
	ord_dic = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa':2, 'Po':1}

	for col in ord_cols:
	    inputDF[col] = inputDF[col].map(lambda x: ord_dic.get(x, 0))
	
	
	############################### Transform numerical data to categorical  ##########
	
	inputDF.MSSubClass = inputDF.MSSubClass.astype('str')
	inputDF.YrSold = inputDF.YrSold.astype('str')
	inputDF.MoSold = inputDF.MoSold.astype('str')
		
	############################### Label frequency or onehot encoding  ##############
	
	##Get all numerical feature transformed
	if isBoxCox:
		numeric_feats = inputDF.dtypes[inputDF.dtypes != "object"].index
		skewed_data = inputDF[numeric_feats].apply(lambda x: skew(x))
		skewed = skewed_data[abs(skewed_data) > 0.75]
		for i in skewed.index:
			inputDF[i] = boxcox1p(inputDF[i], 0.15)
			#print ( i )
	
	##Onehot or label encoding	
	if onehot:
		
		numeric_feats = inputDF.dtypes[inputDF.dtypes != "object"].index
		object_feats = inputDF.dtypes[inputDF.dtypes == "object"].index
		for i, c in enumerate(object_feats):
			if c == skipFeature:
				object_feats.delete(i)
				skipped = inputDF[skipFeature]
				
		objEnc = pd.get_dummies(inputDF[object_feats], drop_first=True, dummy_na=True)
		numEnc = inputDF[numeric_feats]
		
		try:
		    skipped
		except NameError:
		    var_exists = False
		else:
		    var_exists = True
		if var_exists:
			inputDF = pd.concat( [objEnc,numEnc, skipped] , axis=1, join='outer', sort=True)
		else:
			inputDF = pd.concat( [objEnc,numEnc] , axis=1, join='outer', sort=True)
		

	else:		
		for i, c in enumerate ( inputDF.columns ):
			if inputDF[c].dtype == 'object' and c != skipFeature:
				lce = LabelCountEncoder()
				inputDF[c] = lce.fit_transform(inputDF[c])
				encodedDic[inputDF.columns[i]] = lce.rev_count_dict  #add reversed dic back to the global variable
				#print ( c )	


	return [inputDF, encodedDic]

	



















































