# Importing objects, libraries and functions to be used in the algorithm.
import pandas as pd
import numpy as np
import quantopian.algorithm as algo  
import quantopian.experimental.optimize as opt  
from quantopian.pipeline import Pipeline, CustomFactor  
from quantopian.pipeline.data import builtin, morningstar as mstar  
from quantopian.pipeline.factors.morningstar import MarketCap  
from quantopian.pipeline.classifiers.morningstar import Sector  
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.data.builtin import USEquityPricing

# Constraint Parameters.  
MAX_GROSS_LEVERAGE = 1.0  
MAX_SHORT_POSITION_SIZE = 0.0 # 0.0%  
MAX_LONG_POSITION_SIZE = 0.01 # 1.0%  

# Scheduling Parameters. How long to wait before start after the market opens.
MINUTES_AFTER_MARKET = 10

# Momentum is defined as the return of a security over the period of the
# last 11 months with 1-month gap between the end of the 11th month and today.
class Momentum(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, close):
        out[:] = close[-20] / close[0]

# The function initialize is expected to be defined
# by default in any Quantopian algorithm.
def initialize(context):  
    # To set a custom benchmark the following function can be called:
    # set_benchmark(symbol('IWV'))
    # Otherwise the default benchmark will be used (SPY).  
    
    # Commission is set to be $0.005 per share and $1 per trade.  
    set_commission(us_equities=commission.PerShare(cost=0.005, min_trade_cost=1))

    # Exchange code of a firm.
    exchange = mstar.share_class_reference.exchange_id.latest
    
    # A filter rule is created that returns True only for
    # the stocks from the exchanges listed.
    my_exchanges = exchange.element_of(['NYSE','NYS','NAS','ASE'])
    
    # Market capitalisation, sector code and momentum of a firm.
    market_cap = MarketCap()
    sector = Sector()
    umd = Momentum()
    
    # Defining total_equity, operating_income and interest_expense as
    # corresponding values in the latest income statement and balance sheet.
    total_equity = mstar.balance_sheet.total_equity.latest
    operating_income = mstar.income_statement.operating_income.latest
    interest_expense = mstar.income_statement.interest_expense.latest
    
    # The trading universe is defined as QTradableStocksUS that falls into
    # my_exchanges and has data for umd, total_equity, operating_income,
    # interest_expense, market_cap and sector.
    universe_exchange = QTradableStocksUS() & umd.notnull() & my_exchanges & total_equity.notnull() & market_cap.notnull() & sector.notnull() & operating_income.notnull() & interest_expense.notnull()
    
    # Small and large market cap groups specified as percentile.
    small = (MarketCap(mask=universe_exchange).percentile_between(0, 50))
    large = (MarketCap(mask=universe_exchange).percentile_between(50, 100))
    
    # Create a filter that returns True for the assets in the universe
    # that belong to the given sector(s).
    sec = mstar.asset_classification.morningstar_sector_code.latest
    my_sec = sec.element_of([101])
    
    # Here the universe redefined as universe_exchange that belongs
    # to the sector(s) in 'my_sec' and falls into either
    # small or large market cap group as defined above.
    # my_sec should be uncommented in case if a speficic sector is wanted.
    '''
    Here are the sector codes that might be used:
    
     -1: 'Misc',  
    101: 'Basic Materials',  
    102: 'Consumer Cyclical',  
    103: 'Financial Services',  
    104: 'Real Estate',  
    205: 'Consumer Defensive',  
    206: 'Healthcare',  
    207: 'Utilities',  
    308: 'Communication Services',  
    309: 'Energy',  
    310: 'Industrials',  
    311: 'Technology' , 
    '''
    universe = universe_exchange & small #& my_sec 
    
    # Book to market is defined as total_equity divided by the market_cap.
    # The value is normalised and ranked in an ascending order.
    bm = total_equity / market_cap
    bm_weights = bm.rank(ascending=True, mask=universe)
    
    # Operating profitability ratio is defined as operating_income subtracted
    # interest_expense divided by the total_equity.
    # The value is normalised and ranked in an ascending order.
    op = (operating_income - interest_expense) / total_equity
    op_weights = op.rank(ascending=True, mask=universe)
  
    # Price momentum values are ranked and normalised in an ascending order.
    umd_weights = umd.rank(ascending=True, mask=universe)
    
    # A class JoinFactors is defined that is used to combine the normalised
    # scores of the factors defined above.
    class JoinFactors(CustomFactor):  
        #inputs = [factor1, factor2, ...] There can be multiple inputs.
        window_length = 1

        def compute(self, today, assets, out, *inputs):  
            array = np.concatenate(inputs, axis=0)  
            out[:] = np.nansum(array, axis=0)  
            out[ np.all(np.isnan(array), axis=0) ] = np.nan
    
    # window_safe declares that scores of the factors are robust to
    # pricing adjustments from splits or dividends. In other words,
    # the value that will be the same no matter what day you are
    # looking back from. This is a required step in order to
    # use them as the input to JoinFactors.
    bm_weights.window_safe = True  
    op_weights.window_safe = True
    umd_weights.window_safe = True

    # The weights of the combined factor. 1, 2, 3 or more factors can be used.
    final_weights = JoinFactors(inputs=[bm_weights, op_weights, umd_weights], mask=universe)
    universe = final_weights.notnan()        
    
    # The Pipeline object filled with the data defined above is returned.
    pipe = Pipeline(
        columns={
                 'bm_weights': bm_weights,
                 'op_weights': op_weights,
                 'umd_weights': umd_weights,
                 'alpha':final_weights,
                 'exchange': exchange,
                 'market_cap': market_cap,
                 'sector': sector,
                },
        # Screen out all the data points outside the trading universe.
        screen = universe
    )  

    # The function attach_pipeline is called 
    # to load the data in defined in the pipeline.
    algo.attach_pipeline(pipe, 'pipe')  

    # Schedule a function, 'do_portfolio_construction', to run once a month  
    # ten minutes after market is open.  
    algo.schedule_function(  
        do_portfolio_construction,  
        date_rule=algo.date_rules.month_start(),  
        time_rule=algo.time_rules.market_open(minutes=MINUTES_AFTER_MARKET),  
        half_days=False,  
    )  

# The function before_trading_start defines the logic
# that happens every time before the trading session begins.
# Here pipeline output is processed.
def before_trading_start(context, data):  
    context.pipeline_data = algo.pipeline_output('pipe')  

# Portfolio construction. Inside this function the strategy is expressed
# as a set of objectives and constraints.
def do_portfolio_construction(context, data):  
    pipeline_data = context.pipeline_data  
    todays_universe = pipeline_data.index  
 
    # Objective here was to maximise alpha which is 
    # our factor defined in the pipeline.
    objective = opt.MaximizeAlpha(pipeline_data.alpha)  

    # Constrain our gross leverage to 1.0 or less.   
    # This means that the absolute value of our long and short positions 
    # should not exceed the value of our portfolio.  
    constrain_gross_leverage = opt.MaxGrossLeverage(MAX_GROSS_LEVERAGE)  

    # Constrain individual position size to no more than a fixed percentage   
    # of our portfolio.  
    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
        -MAX_SHORT_POSITION_SIZE,
        MAX_LONG_POSITION_SIZE,
    )

    # Constrain ourselves to allocate the same amount of capital to   
    # long and short positions. Not used in the simulations in this work.
    market_neutral = opt.DollarNeutral()  

    # Constrain the maximum average exposure 
    # to individual sectors to -10% - 10%.
    sector_neutral = opt.NetPartitionExposure.with_equal_bounds(
        labels=pipeline_data.sector,
        min=-0.10,
        max=0.10,
    )  

    # Run the optimization. 
    # This will calculate new portfolio weights and  
    # manage moving our portfolio toward the target.  
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=[
            constrain_gross_leverage,
            constrain_pos_size,
            # market_neutral, ---> not used in the study.
            sector_neutral,
        ],
        universe=todays_universe,  
    )