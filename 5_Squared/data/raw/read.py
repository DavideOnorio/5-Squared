import bql

bq = bql.Service()
members = bq.execute(
    bql.Request("SPX Index", bql.data.members(), dates="2018-01-01")
)