import openpyxl
import pandas as pd

def read_excel(filename, nrows):
    """Read out a subset of rows from the first worksheet of an excel workbook.

    This function will not load more excel rows than necessary into memory, and is
    therefore well suited for very large excel files.

    Parameters
    ----------
    filename : str or file-like object
        Path to excel file.
    nrows : int
        Number of rows to parse (starting at the top).

    Returns
    -------
    pd.DataFrame
        Column labels are constructed from the first row of the excel worksheet.
        :param index:

    """
    # Parameter `read_only=True` leads to excel rows only being loaded as-needed
    book = openpyxl.load_workbook(filename=filename, read_only=True, data_only=True)
    first_sheet = book.worksheets[0]
    rows_generator = first_sheet.values

    header_row = next(rows_generator)
    data_rows = [row for (_, row) in zip(range(nrows - 1), rows_generator)]
    return pd.DataFrame(data_rows, columns=header_row)


#

# def plot_dataset(df, title, x):
#     data = []
#     value = go.Scatter(
#         x=df.index,
#         y=df.value,
#         mode="lines",
#         name="values",
#         marker=dict(),
#         text=df.index,
#         line=dict(color="rgba(0,0,0, 0.3)"),
#     )
#     data.append(value)
#
#     layout = dict(
#         title=title,
#         xaxis=dict(title="Date", ticklen=5, zeroline=False),
#         yaxis=dict(title="Value", ticklen=5, zeroline=False),
#     )
#
#     fig = dict(data=data, layout=layout)
#     iplot(fig)

