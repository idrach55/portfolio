from .risk import Driver, Utils


class TestPortfolio:
    def testGetPrices(self):
        # We know these values and assume data remains constant...
        px = Driver.getPrices(['SPY']).loc['2020']
        assert len(px) == 253
        assert px['SPY'].loc['2020-01-02'].round(2) == 307.20
        assert px['SPY'].loc['2020-12-31'].round(2) == 360.27

    def testGetIsCached(self):
        # Assuming tests run sequentially, data will be cached from above.
        assert Driver.getIsCached('SPY')

    def testGetMetrics(self):
        px = Driver.getPrices(['SPY']).loc['2020']
        expected = {
            'tr': 0.1598,
            'vol': 0.3365,
            'sharpe': 0.4749,
            'sortino': 0.4117,
            'maxdraw': 0.337
        }
        metrics = Utils.getMetrics(px)[0].loc['SPY'].round(4)
        assert dict(metrics) == expected