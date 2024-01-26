from IPython.display import HTML, display
from redis.commands.search.result import Result


def table_print(dict_list):
    # If there's nothing in the list, there's nothing to print
    if len(dict_list) == 0:
        return

    # Getting column names (dictionary keys) using the first dictionary
    columns = dict_list[0].keys()

    # HTML table header
    html = "<table><tr><th>"
    html += "</th><th>".join(columns)
    html += "</th></tr>"

    # HTML table content
    for dictionary in dict_list:
        html += "<tr><td>"
        html += "</td><td>".join(str(dictionary[column]) for column in columns)
        html += "</td></tr>"

    # HTML table footer
    html += "</table>"

    # Displaying the table
    display(HTML(html))


def result_print(results):
    if isinstance(results, Result):
        # If there's nothing in the list, there's nothing to print
        if len(results.docs) == 0:
            return

        results = [doc.__dict__ for doc in results.docs]

    to_remove = ["id", "payload"]
    for doc in results:
        for key in to_remove:
            if key in doc:
                del doc[key]

    table_print(results)