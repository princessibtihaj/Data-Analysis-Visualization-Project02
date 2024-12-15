'''categorical_analysis.py
Run analyses with categorical data
Princess Ibtihaj 
CS 251: Data Analysis and Visualization
Fall 2024
'''
import numpy as np

import analysis


class CatAnalysis(analysis.Analysis):
    def __init__(self, data):
        '''CatAnalysis constructor

        (This method is provided to you and should not require modification)

        Parameters:
        -----------
        data: `CatData`.
            `CatData` object that stores the dataset.
        '''
        super().__init__(data)

    def cat_count(self, header):
        '''Counts the number of samples that have each level of the categorical variable named `header`

        Example:
            Column of self.data for `cat_var1`: [0, 1, 2, 0, 0, 1, 0, 0]
            This method should return `counts` = [5, 2, 1].

        Parameters:
        -----------
        header: str. Header of the categorical variable whose levels should be returned.

        Returns:
        -----------
        ndarray. shape=(num_levels,). The number of samples that have each level of the categorical variable named `header`
        list of strs. len=num_levels. The level strings of the categorical variable  `header` associated with the counts.

        NOTE:
        - Your implementation should rely on logical indexing. Using np.unique is not allowed here.
        - A single loop over levels is totally fine here.
        - `self.data` stores categorical levels as INTS so it is helpful to work with INT-coded levels when doing the counting.
        The method should, however, return the STRING-coded levels (e.g. for plotting).
        '''
        column = self.data.select_data([header])
        str_levels = self.data.get_cat_levels_str(header)
        int_levels = self.data.get_cat_levels_int(header)
        counts = [np.sum(column == int_level) for int_level in int_levels]
        return counts, str_levels

    def cat_mean(self, numeric_header, categorical_header):
        '''Computes the mean of values of the numeric variable `numeric_header` for each of the different categorical
        levels of the variable `categorical_header`.

        POSSIBLE EXTENSION. NOT REQUIRED FOR BASE PROJECT

        Example:
            Column of self.data for `numeric_var1` = [4, 5, 6, 1, 2, 3]
            Column of self.data for `cat_var1` = [0, 0, 0, 1, 1, 1]

            If `numeric_header` = "numeric_var1" and `categorical_header` = "cat_var1", this method should return
            `means` = [5, 2].
            (1st entry is mean of all numeric var values with corresponding int level of 0,
             2nd entry is mean of all numeric var values with corresponding int level of 1)

        Parameters:
        -----------
        numeric_header: str. Header of the numeric variable whose values should be averaged.
        categorical_header: str. Header of the categorical variable whose levels determine which values of the
            numeric variable that should be averaged.

        Returns:
        -----------
        ndarray. shape=(num_levels,). Means of values of the numeric variable `numeric_header` for each of the different
            categorical levels of the variable `categorical_header`.
        list of strs. len=num_levels. The level strings of the categorical variable  `categorical_header` associated with
            the counts.

        NOTE:
        - Your implementation should rely on logical indexing. Using np.unique is not allowed here.
        - A single loop over levels is totally fine here.
        - As above, it is easier to work with INT-coded levels, but the STRING-coded levels should be returned.
        - Since your numeric data has nans in it, you should use np.nanmean, which ignores any nan values. Otherwise, the
        according to np.mean, the mean of any collection of numbers that include at least one nan will always be nan.
        You can easily swap np.mean with np.nanmean: https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html
        '''
        numeric_column = self.data.select_data([numeric_header])
        categorical_column = self.data.select_data([categorical_header])
        str_levels = self.data.get_cat_levels_str(categorical_header)
        int_levels = self.data.get_cat_levels_int(categorical_header)
        
        means = []
        for int_level in int_levels:
            mask = (categorical_column == int_level)
            level_mean = np.nanmean(numeric_column[mask])
            means.append(level_mean)
        
        return np.array(means), str_levels

    def cat_count2(self, header1, header2):
        '''Counts the number of samples that have all combinations of levels coming from two categorical headers
        (`header1` and `header2`).

        POSSIBLE EXTENSION. NOT REQUIRED FOR BASE PROJECT

        Parameters:
        -----------
        header1: str. Header of the first categorical variable
        header2: str. Header of the second categorical variable

        Returns:
        -----------
        ndarray. shape=(header1_num_levels, header2_num_levels). The number of samples that have each combination of
            levels of the categorical variables `header1` and `header2`.
        list of strs. len=header1_num_levels. The level strings of the categorical variable  `header1`
        list of strs. len=header2_num_levels. The level strings of the categorical variable  `header2`

        Example:

        header1_level_strs: ['a', 'b']
        header2_level_strs: ['y', 'z']

        counts =
                [num samples with header1 value 'a' AND header2 value 'y', num samples with header1 value 'a' AND header2 value 'z']
                [num samples with header1 value 'b' AND header2 value 'y', num samples with header1 value 'b' AND header2 value 'z']

        NOTE:
        - To combine two logical arrays element-wise, you can use the & operator or np.logical_and
        '''
        column1 = self.data.select_data([header1])
        column2 = self.data.select_data([header2])
        str_levels1 = self.data.get_cat_levels_str(header1)
        int_levels1 = self.data.get_cat_levels_int(header1)
        str_levels2 = self.data.get_cat_levels_str(header2)
        int_levels2 = self.data.get_cat_levels_int(header2)
        
        count_matrix = np.zeros((len(int_levels1), len(int_levels2)))
        
        for i, int_level1 in enumerate(int_levels1):
            for j, int_level2 in enumerate(int_levels2):
                mask = (column1 == int_level1) & (column2 == int_level2)
                count_matrix[i, j] = np.sum(mask)
                
        return count_matrix, str_levels1, str_levels2
