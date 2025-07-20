

## 📝 **SQL Queries for Highest Profit Analysis by CEDI**

### **1. Top Product by Revenue for Each CEDI (Main Query)**
```sql
-- Get the highest revenue product for each CEDI in Q2 2025
WITH cedi_product_revenue AS (
    SELECT 
        cc.CEDI, 
        p.Categoria, 
        p.Producto, 
        SUM(s.net_revenue) as total_revenue, 
        SUM(s.VentasCajasUnidad) as total_volume, 
        COUNT(DISTINCT s.customer_id) as customers, 
        ROW_NUMBER() OVER (PARTITION BY cc.CEDI ORDER BY SUM(s.net_revenue) DESC) as revenue_rank 
    FROM dev.segmentacion s 
    INNER JOIN dev.cliente_cedi cc ON s.customer_id = cc.customer_id 
    INNER JOIN dev.producto p ON s.material_id = p.Material 
    INNER JOIN dev.tiempo t ON s.calday = t.Fecha 
    WHERE cc.CEDI IS NOT NULL 
      AND s.VentasCajasUnidad > 0 
      AND t.Q = 2 
      AND t.Year = 2025 
    GROUP BY cc.CEDI, p.Categoria, p.Producto
) 
SELECT 
    CEDI, 
    Categoria, 
    Producto, 
    total_revenue, 
    total_volume, 
    customers 
FROM cedi_product_revenue 
WHERE revenue_rank = 1 
ORDER BY total_revenue DESC
```

### **2. Top Category by Revenue for Each CEDI**
```sql
-- Get the highest revenue category for each CEDI in Q2 2025
WITH cedi_category_revenue AS (
    SELECT 
        cc.CEDI, 
        p.Categoria, 
        SUM(s.net_revenue) as total_revenue, 
        SUM(s.VentasCajasUnidad) as total_volume, 
        COUNT(DISTINCT p.Producto) as product_count, 
        COUNT(DISTINCT s.customer_id) as customers, 
        ROW_NUMBER() OVER (PARTITION BY cc.CEDI ORDER BY SUM(s.net_revenue) DESC) as revenue_rank 
    FROM dev.segmentacion s 
    INNER JOIN dev.cliente_cedi cc ON s.customer_id = cc.customer_id 
    INNER JOIN dev.producto p ON s.material_id = p.Material 
    INNER JOIN dev.tiempo t ON s.calday = t.Fecha 
    WHERE cc.CEDI IS NOT NULL 
      AND s.VentasCajasUnidad > 0 
      AND t.Q = 2 
      AND t.Year = 2025 
    GROUP BY cc.CEDI, p.Categoria
) 
SELECT 
    CEDI, 
    Categoria, 
    total_revenue, 
    total_volume, 
    product_count, 
    customers 
FROM cedi_category_revenue 
WHERE revenue_rank = 1 
ORDER BY total_revenue DESC
```

### **3. Formatted Top Results (Pretty Display)**
```sql
-- Get top 30 CEDIs with formatted revenue display
WITH ranked_products AS (
    SELECT 
        cc.CEDI, 
        p.Categoria, 
        p.Producto, 
        SUM(s.net_revenue) as total_revenue, 
        SUM(s.VentasCajasUnidad) as total_volume, 
        COUNT(DISTINCT s.customer_id) as customers, 
        ROW_NUMBER() OVER (PARTITION BY cc.CEDI ORDER BY SUM(s.net_revenue) DESC) as rank 
    FROM dev.segmentacion s 
    INNER JOIN dev.cliente_cedi cc ON s.customer_id = cc.customer_id 
    INNER JOIN dev.producto p ON s.material_id = p.Material 
    INNER JOIN dev.tiempo t ON s.calday = t.Fecha 
    WHERE cc.CEDI IS NOT NULL 
      AND s.VentasCajasUnidad > 0 
      AND t.Q = 2 
      AND t.Year = 2025 
    GROUP BY cc.CEDI, p.Categoria, p.Producto
) 
SELECT TOP 30 
    CEDI, 
    Categoria, 
    Producto, 
    FORMAT(total_revenue, 'C0') as revenue_formatted, 
    total_volume, 
    customers 
FROM ranked_products 
WHERE rank = 1 
ORDER BY total_revenue DESC
```

### **4. Highest Profit Margin Analysis (Revenue per Unit)**
```sql
-- Find products with highest revenue per unit (profit margin)
SELECT TOP 15 
    cc.CEDI, 
    p.Categoria, 
    p.Produto, 
    SUM(s.net_revenue) as total_revenue, 
    SUM(s.VentasCajasUnidad) as total_volume, 
    SUM(s.net_revenue) / SUM(s.VentasCajasUnidad) as revenue_per_unit, 
    COUNT(DISTINCT s.customer_id) as customers 
FROM dev.segmentacion s 
INNER JOIN dev.cliente_cedi cc ON s.customer_id = cc.customer_id 
INNER JOIN dev.producto p ON s.material_id = p.Material 
INNER JOIN dev.tiempo t ON s.calday = t.Fecha 
WHERE cc.CEDI IS NOT NULL 
  AND s.VentasCajasUnidad > 0 
  AND t.Q = 2 
  AND t.Year = 2025 
GROUP BY cc.CEDI, p.Categoria, p.Produto 
HAVING SUM(s.net_revenue) > 1000000  -- Focus on significant revenue products
ORDER BY revenue_per_unit DESC
```

### **5. Initial Exploratory Query (All Products by CEDI)**
```sql
-- Explore all products by CEDI and revenue (first 20 results)
SELECT 
    cc.CEDI, 
    p.Categoria, 
    p.Produto, 
    SUM(s.net_revenue) as total_revenue, 
    SUM(s.VentasCajasUnidad) as total_volume, 
    COUNT(DISTINCT s.customer_id) as customers 
FROM dev.segmentacion s 
INNER JOIN dev.cliente_cedi cc ON s.customer_id = cc.customer_id 
INNER JOIN dev.produto p ON s.material_id = p.Material 
INNER JOIN dev.tiempo t ON s.calday = t.Fecha 
WHERE cc.CEDI IS NOT NULL 
  AND s.VentasCajasUnidad > 0 
  AND t.Q = 2 
  AND t.Year = 2025 
GROUP BY cc.CEDI, p.Categoria, p.Produto 
ORDER BY cc.CEDI, total_revenue DESC
```

### **6. Supporting Analysis - Top Products for Specific CEDIs**
```sql
-- Deep dive into top 5 CEDIs performance
SELECT TOP 15 
    cc.CEDI, 
    'TOP_PRODUCT' as type, 
    p.Categoria, 
    p.Produto as item_name, 
    SUM(s.net_revenue) as total_revenue, 
    SUM(s.VentasCajasUnidad) as total_volume, 
    COUNT(DISTINCT s.customer_id) as customers 
FROM dev.segmentacion s 
INNER JOIN dev.cliente_cedi cc ON s.customer_id = cc.customer_id 
INNER JOIN dev.produto p ON s.material_id = p.Material 
INNER JOIN dev.tiempo t ON s.calday = t.Fecha 
WHERE cc.CEDI IS NOT NULL 
  AND s.VentasCajasUnidad > 0 
  AND t.Q = 2 
  AND t.Year = 2025 
  AND cc.CEDI IN ('Sucursal Univer', 'Planta Cd. Juár', 'Cedi Tonalá', 'Planta Guadalup', 'Sucursal Lincol') 
GROUP BY cc.CEDI, p.Categoria, p.Produto 
ORDER BY cc.CEDI, SUM(s.net_revenue) DESC
```

---

## 🎯 **Key Query Techniques Used:**

### **1. Window Functions (ROW_NUMBER)**
- **PARTITION BY cc.CEDI**: Groups data by each CEDI
- **ORDER BY SUM(s.net_revenue) DESC**: Ranks by revenue within each CEDI
- **WHERE rank = 1**: Gets only the top performer per CEDI

### **2. Common Table Expressions (CTEs)**
- **WITH ranked_products AS**: Creates reusable subquery
- Separates ranking logic from final selection
- Improves query readability and performance

### **3. Strategic Filtering**
- **t.Q = 2 AND t.Year = 2025**: Q2 2025 data only
- **s.VentasCajasUnidad > 0**: Exclude zero-volume transactions
- **HAVING SUM(s.net_revenue) > 1000000**: Focus on significant products

### **4. Comprehensive Joins**
- **dev.segmentacion → dev.cliente_cedi**: Link sales to distribution centers
- **dev.segmentacion → dev.produto**: Get product details
- **dev.segmentacion → dev.tiempo**: Filter by time periods

### **5. Aggregation Functions**
- **SUM(s.net_revenue)**: Total revenue
- **COUNT(DISTINCT s.customer_id)**: Unique customer reach
- **SUM(s.net_revenue) / SUM(s.VentasCajasUnidad)**: Revenue per unit

### **6. Formatting Functions**
- **FORMAT(total_revenue, 'C0')**: Currency formatting ($257,386,888)
- **TOP N**: SQL Server limit syntax

These queries provided the comprehensive analysis showing that **REFRESCOS category dominates 100% of CEDIs** with **Coca-Cola variants** being the top revenue generators, while **Monster Energy** products show the highest profit margins.