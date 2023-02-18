import pandas as pd
import numpy as np


def error_type_row(df: pd.DataFrame, my_label: str = "computed_label", answer_label: str = "Label") -> tuple[pd.DataFrame, pd.DataFrame]:
    # df = df.copy()
    # print(df.columns)
    # print(df["computed_label"])
    # print(df[my_label])
    def helper(mine, theirs):
        if theirs == mine:
            return "Correct Answer"
        elif bool(theirs) and not bool(mine):
            return "False Negative"
        else:
            return "False Positive"
    df["error_type"] = df.apply(lambda x: helper(x[my_label], x[answer_label]), axis=1)

    return df, df.loc[(df['error_type'] == 'False Positive') | (df['error_type'] == 'False Negative')] #df[df["error_type"].isin(("False Negative", "False Positive"))]




