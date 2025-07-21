
# **🔧 INSTRUCTIONS FOR ADVANCED ANALYTICS**

## **📍 OPERATIONAL INTELLIGENCE**

### **Route Performance and Coverage**

**Available Data:**
- Customer-Territory assignments via `cliente_cedi`
- Coverage metrics (94-95% in top territories)
- Geographic hierarchy: Region → Territorio → Subterritorio

**MCP Query Patterns:**
```sql
-- Route Coverage Analysis
WITH RouteCoverage AS (
    SELECT 
        cc.Region,
        cc.Territorio,
        cc.CEDI,
        COUNT(DISTINCT cc.customer_id) as total_customers,
        COUNT(DISTINCT s.customer_id) as active_customers,
        COUNT(DISTINCT s.calday) as service_days,
        ROUND(COUNT(DISTINCT s.customer_id) * 100.0 / COUNT(DISTINCT cc.customer_id), 2) as coverage_pct
    FROM dev.cliente_cedi cc
    LEFT JOIN dev.segmentacion s ON cc.customer_id = s.customer_id 
        AND s.calday >= '2025-04-01' AND s.VentasCajasUnidad > 0
    WHERE cc.Region IS NOT NULL
    GROUP BY cc.Region, cc.Territorio, cc.CEDI
)
SELECT * FROM RouteCoverage 
WHERE coverage_pct < 90  -- Identify underperforming routes
ORDER BY coverage_pct ASC;
```

**Limitations & Workarounds:**
❌ **No direct route data** - Use Territory as route proxy
❌ **No delivery times** - Use transaction frequency as proxy
✅ **Coverage calculation**: Active customers / Total customers
✅ **Performance metrics**: Volume per territory, service frequency

### **Territory Analysis and Optimization**

**Available Metrics:**
- 25 territories across 4 regions
- Customer density per territory (7K-35K customers)
- Service coverage (88-95% range)

**MCP Implementation:**
```sql
-- Territory Performance Matrix
WITH TerritoryMetrics AS (
    SELECT 
        cc.Region,
        cc.Territorio,
        COUNT(DISTINCT cc.customer_id) as customer_base,
        SUM(s.VentasCajasUnidad) as total_volume,
        SUM(s.net_revenue) as total_revenue,
        COUNT(DISTINCT cc.CEDI) as cedis_serving,
        ROUND(AVG(s.VentasCajasUnidad), 2) as avg_transaction_size
    FROM dev.cliente_cedi cc
    INNER JOIN dev.segmentacion s ON cc.customer_id = s.customer_id
    WHERE s.calday >= '2025-04-01' AND s.VentasCajasUnidad > 0
    GROUP BY cc.Region, cc.Territorio
)
SELECT 
    *,
    ROUND(total_revenue / customer_base, 2) as revenue_per_customer,
    ROW_NUMBER() OVER (PARTITION BY Region ORDER BY total_revenue DESC) as regional_rank
FROM TerritoryMetrics
ORDER BY total_revenue DESC;
```

### **CEDI Dispatch Volumes**

**Available Data:**
- 119 unique CEDIs
- Volume aggregation capability
- Customer assignments per CEDI

**MCP Query Pattern:**
```sql
-- CEDI Dispatch Volume Analysis
SELECT 
    cc.CEDI,
    cc.Region,
    COUNT(DISTINCT cc.customer_id) as customers_served,
    SUM(s.VentasCajasUnidad) as total_dispatch_volume,
    SUM(s.net_revenue) as total_revenue,
    COUNT(DISTINCT s.calday) as operational_days,
    ROUND(SUM(s.VentasCajasUnidad) / COUNT(DISTINCT s.calday), 2) as avg_daily_volume,
    ROUND(SUM(s.net_revenue) / COUNT(DISTINCT s.calday), 2) as avg_daily_revenue
FROM dev.cliente_cedi cc
INNER JOIN dev.segmentacion s ON cc.customer_id = s.customer_id
WHERE s.calday >= '2025-04-01' AND s.VentasCajasUnidad > 0
AND cc.CEDI IS NOT NULL
GROUP BY cc.CEDI, cc.Region
ORDER BY total_dispatch_volume DESC;
```

### **Inventory Turnover Rates**

**Available Proxies:**
- Daily transaction patterns
- Product movement frequency
- Volume consistency

**MCP Implementation:**
```sql
-- Product Turnover Analysis by CEDI
WITH ProductTurnover AS (
    SELECT 
        cc.CEDI,
        p.Material,
        p.Producto,
        p.Categoria,
        COUNT(DISTINCT s.calday) as sales_days,
        SUM(s.VentasCajasUnidad) as total_moved,
        ROUND(AVG(s.VentasCajasUnidad), 2) as avg_daily_movement,
        STDDEV(s.VentasCajasUnidad) as movement_volatility
    FROM dev.segmentacion s
    INNER JOIN dev.cliente_cedi cc ON s.customer_id = cc.customer_id
    INNER JOIN dev.producto p ON s.material_id = p.Material
    WHERE s.calday >= '2025-04-01' AND s.VentasCajasUnidad > 0
    GROUP BY cc.CEDI, p.Material, p.Producto, p.Categoria
    HAVING COUNT(DISTINCT s.calday) >= 30  -- Consistent movers
)
SELECT 
    CEDI,
    Categoria,
    COUNT(*) as active_products,
    ROUND(AVG(avg_daily_movement), 2) as category_avg_turnover,
    SUM(total_moved) as category_total_volume
FROM ProductTurnover
GROUP BY CEDI, Categoria
ORDER BY CEDI, category_avg_turnover DESC;
```

### **Sales Force Effectiveness**

**Available Proxies:**
- Customer coverage per territory
- Revenue per customer metrics
- Service consistency

**MCP Pattern:**
```sql
-- Sales Force Effectiveness Proxy
WITH SalesEffectiveness AS (
    SELECT 
        cc.Region,
        cc.Territorio,
        COUNT(DISTINCT cc.customer_id) as territory_customers,
        COUNT(DISTINCT s.customer_id) as customers_reached,
        COUNT(DISTINCT s.calday) as active_service_days,
        SUM(s.net_revenue) as territory_revenue,
        COUNT(DISTINCT cc.distribution_channel_id) as channel_diversity
    FROM dev.cliente_cedi cc
    LEFT JOIN dev.segmentacion s ON cc.customer_id = s.customer_id 
        AND s.calday >= '2025-04-01'
    WHERE cc.Region IS NOT NULL
    GROUP BY cc.Region, cc.Territorio
)
SELECT 
    *,
    ROUND(customers_reached * 100.0 / territory_customers, 2) as reach_effectiveness,
    ROUND(territory_revenue / NULLIF(customers_reached, 0), 2) as revenue_per_customer,
    ROUND(territory_revenue / NULLIF(active_service_days, 0), 2) as daily_productivity
FROM SalesEffectiveness
ORDER BY reach_effectiveness DESC;
```

---

## **🎯 PRODUCT & MARKETING**

### **New Product Adoption Rates**

**Identification Method:**
- Products appearing after specific launch dates
- Customer acquisition tracking
- Volume growth analysis

**MCP Implementation:**
```sql
-- New Product Adoption Analysis
WITH NewProductLaunches AS (
    SELECT 
        p.Material,
        p.Producto,
        p.Categoria,
        p.AgrupadordeMarca,
        MIN(s.calday) as launch_date,
        COUNT(DISTINCT s.customer_id) as adopting_customers,
        SUM(s.VentasCajasUnidad) as total_adoption_volume,
        COUNT(DISTINCT s.calday) as market_days
    FROM dev.segmentacion s
    INNER JOIN dev.producto p ON s.material_id = p.Material
    WHERE s.VentasCajasUnidad > 0
    GROUP BY p.Material, p.Produto, p.Categoria, p.AgrupadordeMarca
    HAVING MIN(s.calday) >= '2025-06-01'  -- Recent launches
), AdoptionProgress AS (
    SELECT 
        npl.*,
        ROUND(adopting_customers * 100.0 / (
            SELECT COUNT(DISTINCT customer_id) FROM dev.segmentacion 
            WHERE calday >= npl.launch_date
        ), 2) as market_penetration_pct,
        ROUND(total_adoption_volume / market_days, 2) as avg_daily_adoption
    FROM NewProductLaunches npl
)
SELECT * FROM AdoptionProgress
ORDER BY market_penetration_pct DESC;
```

### **Seasonal Trend Analysis**

**Available Patterns:**
- Day-of-week variations (Sunday: 11.0 avg volume vs weekdays: 3.3)
- Monthly progression patterns
- Product category seasonality

**MCP Query:**
```sql
-- Seasonal Pattern Detection
WITH SeasonalPatterns AS (
    SELECT 
        p.Categoria,
        p.AgrupadordeMarca,
        MONTH(s.calday) as month,
        DATEPART(WEEKDAY, s.calday) as day_of_week,
        AVG(s.VentasCajasUnidad) as avg_volume,
        STDDEV(s.VentasCajasUnidad) as volume_volatility,
        COUNT(*) as transaction_count
    FROM dev.segmentacion s
    INNER JOIN dev.producto p ON s.material_id = p.Material
    WHERE s.VentasCajasUnidad > 0
    GROUP BY p.Categoria, p.AgrupadordeMarca, MONTH(s.calday), DATEPART(WEEKDAY, s.calday)
)
SELECT 
    Categoria,
    AgrupadordeMarca,
    month,
    AVG(avg_volume) as monthly_avg_volume,
    MAX(avg_volume) - MIN(avg_volume) as weekly_volatility,
    SUM(transaction_count) as monthly_transactions
FROM SeasonalPatterns
GROUP BY Categoria, AgrupadordeMarca, month
ORDER BY weekly_volatility DESC;
```

### **Promotional Effectiveness**

**Identification:**
- "Obsequio" products in catalog (6 products identified)
- Volume/revenue impact analysis
- Channel effectiveness comparison

**MCP Pattern:**
```sql
-- Promotional Impact Analysis
WITH PromotionalProducts AS (
    SELECT p.Material
    FROM dev.producto p
    WHERE LOWER(p.Produto) LIKE '%obsequio%'
), PromotionalImpact AS (
    SELECT 
        cc.Region,
        cc.distribution_channel_id,
        p.Categoria,
        p.AgrupadordeMarca,
        SUM(CASE WHEN pp.Material IS NOT NULL THEN s.VentasCajasUnidad ELSE 0 END) as promo_volume,
        SUM(CASE WHEN pp.Material IS NULL THEN s.VentasCajasUnidad ELSE 0 END) as regular_volume,
        SUM(CASE WHEN pp.Material IS NOT NULL THEN s.net_revenue ELSE 0 END) as promo_revenue,
        SUM(CASE WHEN pp.Material IS NULL THEN s.net_revenue ELSE 0 END) as regular_revenue
    FROM dev.segmentacion s
    INNER JOIN dev.cliente_cedi cc ON s.customer_id = cc.customer_id
    INNER JOIN dev.produto p ON s.material_id = p.Material
    LEFT JOIN PromotionalProducts pp ON s.material_id = pp.Material
    WHERE s.VentasCajasUnidad > 0 AND cc.Region IS NOT NULL
    GROUP BY cc.Region, cc.distribution_channel_id, p.Categoria, p.AgrupadordeMarca
)
SELECT 
    Region,
    distribution_channel_id,
    Categoria,
    ROUND(promo_volume * 100.0 / NULLIF(promo_volume + regular_volume, 0), 2) as promo_volume_share,
    ROUND(promo_revenue * 100.0 / NULLIF(promo_revenue + regular_revenue, 0), 2) as promo_revenue_share,
    ROUND(promo_revenue / NULLIF(promo_volume, 0), 2) as promo_unit_value,
    ROUND(regular_revenue / NULLIF(regular_volume, 0), 2) as regular_unit_value
FROM PromotionalImpact
WHERE promo_volume > 0
ORDER BY promo_volume_share DESC;
```

### **Brand Penetration Analysis**

**Available Metrics:**
- Brand performance by territory
- Category dominance analysis
- Market share calculations

**MCP Implementation:**
```sql
-- Brand Penetration by Territory
WITH BrandTerritoryShare AS (
    SELECT 
        cc.Region,
        cc.Territorio,
        p.Categoria,
        p.AgrupadordeMarca,
        SUM(s.VentasCajasUnidad) as brand_volume,
        SUM(s.net_revenue) as brand_revenue,
        COUNT(DISTINCT s.customer_id) as brand_customers
    FROM dev.segmentacion s
    INNER JOIN dev.cliente_cedi cc ON s.customer_id = cc.customer_id
    INNER JOIN dev.produto p ON s.material_id = p.Material
    WHERE s.VentasCajasUnidad > 0 AND cc.Region IS NOT NULL
    GROUP BY cc.Region, cc.Territorio, p.Categoria, p.AgrupadordeMarca
), TerritoryTotals AS (
    SELECT 
        Region,
        Territorio,
        Categoria,
        SUM(brand_volume) as category_total_volume,
        SUM(brand_revenue) as category_total_revenue,
        COUNT(DISTINCT brand_customers) as category_total_customers
    FROM BrandTerritoryShare
    GROUP BY Region, Territorio, Categoria
)
SELECT 
    bts.Region,
    bts.Territorio,
    bts.Categoria,
    bts.AgrupadordeMarca,
    ROUND(bts.brand_volume * 100.0 / tt.category_total_volume, 2) as volume_share_pct,
    ROUND(bts.brand_revenue * 100.0 / tt.category_total_revenue, 2) as revenue_share_pct,
    ROUND(bts.brand_customers * 100.0 / tt.category_total_customers, 2) as customer_share_pct
FROM BrandTerritoryShare bts
INNER JOIN TerritoryTotals tt ON bts.Region = tt.Region 
    AND bts.Territorio = tt.Territorio 
    AND bts.Categoria = tt.Categoria
ORDER BY bts.Region, bts.Territorio, volume_share_pct DESC;
```

---

## **📊 FORECASTING & PLANNING**

### **Sales Forecasting (Product/CEDI/Territory)**

**Method:** Time series with growth rate projection

**MCP Pattern:**
```sql
-- Sales Forecasting Model
WITH MonthlyTrends AS (
    SELECT 
        cc.CEDI,
        cc.Region,
        p.Material,
        p.Categoria,
        YEAR(s.calday) as year,
        MONTH(s.calday) as month,
        SUM(s.VentasCajasUnidad) as monthly_volume,
        SUM(s.net_revenue) as monthly_revenue
    FROM dev.segmentacion s
    INNER JOIN dev.cliente_cedi cc ON s.customer_id = cc.customer_id
    INNER JOIN dev.produto p ON s.material_id = p.Material
    WHERE s.VentasCajasUnidad > 0
    GROUP BY cc.CEDI, cc.Region, p.Material, p.Categoria, YEAR(s.calday), MONTH(s.calday)
), TrendAnalysis AS (
    SELECT 
        CEDI,
        Region,
        Material,
        Categoria,
        year,
        month,
        monthly_volume,
        monthly_revenue,
        LAG(monthly_volume, 1) OVER (PARTITION BY CEDI, Material ORDER BY year, month) as prev_month_volume,
        LAG(monthly_volume, 3) OVER (PARTITION BY CEDI, Material ORDER BY year, month) as three_months_ago,
        AVG(monthly_volume) OVER (PARTITION BY CEDI, Material ORDER BY year, month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as three_month_avg
    FROM MonthlyTrends
)
SELECT 
    CEDI,
    Region,
    Material,
    Categoria,
    year,
    month,
    monthly_volume,
    prev_month_volume,
    three_month_avg,
    CASE 
        WHEN prev_month_volume > 0 
        THEN ROUND((monthly_volume - prev_month_volume) * 100.0 / prev_month_volume, 2)
        ELSE NULL 
    END as mom_growth_rate,
    -- Simple forecast: 3-month average + growth trend
    ROUND(three_month_avg * 1.1, 2) as next_month_forecast
FROM TrendAnalysis
WHERE year = 2025 AND month = 6  -- Latest available data
ORDER BY CEDI, monthly_volume DESC;
```

### **Market Expansion Opportunities**

**Identification:** Low coverage + high potential territories

**MCP Implementation:**
```sql
-- Market Expansion Analysis
WITH TerritoryOpportunity AS (
    SELECT 
        cc.Region,
        cc.Territorio,
        COUNT(DISTINCT cc.customer_id) as total_customers,
        COUNT(DISTINCT s.customer_id) as active_customers,
        SUM(s.net_revenue) as current_revenue,
        COUNT(DISTINCT cc.CEDI) as serving_cedis,
        ROUND(COUNT(DISTINCT s.customer_id) * 100.0 / COUNT(DISTINCT cc.customer_id), 2) as current_coverage_pct
    FROM dev.cliente_cedi cc
    LEFT JOIN dev.segmentacion s ON cc.customer_id = s.customer_id 
        AND s.calday >= '2025-04-01' AND s.VentasCajasUnidad > 0
    WHERE cc.Region IS NOT NULL
    GROUP BY cc.Region, cc.Territorio
), ExpansionPotential AS (
    SELECT 
        *,
        total_customers - active_customers as untapped_customers,
        ROUND(current_revenue / NULLIF(active_customers, 0), 2) as revenue_per_customer,
        ROUND((total_customers - active_customers) * 
              (current_revenue / NULLIF(active_customers, 0)), 2) as potential_additional_revenue
    FROM TerritoryOpportunity
)
SELECT 
    Region,
    Territorio,
    total_customers,
    current_coverage_pct,
    untapped_customers,
    potential_additional_revenue,
    revenue_per_customer,
    serving_cedis,
    -- Opportunity Score: Large untapped base + proven revenue model
    ROUND((untapped_customers * revenue_per_customer) / 1000000, 2) as opportunity_score_millions
FROM ExpansionPotential
WHERE current_coverage_pct < 95  -- Focus on improvable territories
ORDER BY opportunity_score_millions DESC;
```


### **🚨 CRITICAL LIMITATIONS TO COMMUNICATE**

When questions fall into these categories, provide specific limitation responses:

```markdown
## OUT-OF-SCOPE RESPONSES

**Route/Delivery Questions:**
"Our system tracks territory and customer coverage but doesn't have direct route optimization data. I can provide territory performance metrics and customer coverage analysis as alternatives."

**Inventory Management:**
"We can analyze product turnover patterns and movement frequency, but don't have actual inventory stock levels. I can provide turnover analysis and demand patterns instead."

**Marketing ROI:**
"We can track promotional product performance and brand penetration, but don't have marketing spend data for ROI calculations. I can provide promotional effectiveness and market share analysis."

**Advanced Forecasting:**
"We provide trend-based forecasting using historical patterns but don't include external factors like economic indicators or competitive actions."
```
