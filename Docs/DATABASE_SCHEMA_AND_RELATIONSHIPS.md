# Database Schema and Relationships Analysis

## Database Information
- **Server**: tcp:agenticdbsv.database.windows.net
- **Database**: SalesMX02
- **Version**: Microsoft SQL Azure (RTM)
- **Schema**: dev
- **Tables**: 6 tables, 1 view
- **Status**: ✅ Connected

## Tables Overview

### 1. **dev.cliente** (Customer Master Data)
**Purpose**: Customer dimension table with detailed customer information
- **Primary Key**: `customer_id` (varchar(56))
- **Total Columns**: 41 columns
- **Key Fields**:
  - `customer_id` - Unique customer identifier
  - `Nombre_cliente` - Customer name
  - `Canal_Comercial` - Commercial channel
  - `Territorio_del_cliente` - Customer territory
  - `Region` (via cliente_cedi join)

### 2. **dev.segmentacion** (Sales/Revenue Fact Table)
**Purpose**: Main fact table containing sales transactions and revenue data
- **Composite Key**: `customer_id`, `calday`, `material_id`, `distribution_channel_id`
- **Total Columns**: 12 columns
- **Key Fields**:
  - `customer_id` (varchar(20)) - Links to cliente
  - `material_id` (varchar(36)) - Links to producto
  - `calday` (date) - Transaction date
  - `CALMONTH` (int) - Calendar month
  - `VentasCajasUnidad` (float) - Units sold
  - `IngresoNetoSImpuestos` (float) - Net revenue without taxes
  - `net_revenue` (float) - Net revenue
  - `bottles_sold_m` (float) - Bottles sold
  - `VentasCajasOriginales` (float) - Original cases sold
  - `Cobertura` (int) - Coverage metric

### 3. **dev.producto** (Product Master Data)
**Purpose**: Product dimension table with product details
- **Primary Key**: `Material` (varchar(36))
- **Total Columns**: 12 columns
- **Key Fields**:
  - `Material` - Product material code
  - `Producto` - Product name
  - `Categoria` - Product category
  - `Subcategoria` - Product subcategory
  - `AgrupadordeMarca` - Brand grouping
  - `SaborGlobal` - Global flavor
  - `Contenido` - Content/volume
  - `TipodeEmpaque` - Package type

### 4. **dev.cliente_cedi** (Customer Distribution Mapping)
**Purpose**: Bridge table linking customers to distribution centers
- **Composite Key**: `customer_id`, `distribution_channel_id`
- **Total Columns**: 9 columns
- **Key Fields**:
  - `customer_id` (varchar(22)) - Links to cliente
  - `cedi_id` (varchar(22)) - Distribution center ID
  - `distribution_channel_id` (varchar(4)) - Distribution channel
  - `Region` - Geographic region
  - `Territorio` - Territory
  - `Subterritorio` - Sub-territory
  - `LocalForaneo` - Local/External classification

### 5. **dev.mercado** (Market/Territory Data)
**Purpose**: Market and territory dimension table
- **Primary Key**: `CEDIid` (varchar(8))
- **Total Columns**: 6 columns
- **Key Fields**:
  - `CEDIid` - Distribution center ID
  - `CEDI` - Distribution center name
  - `Zona` - Zone
  - `Territorio` - Territory
  - `Subterritorio` - Sub-territory
  - `LocalForaneo` - Local/External classification

### 6. **dev.tiempo** (Time Dimension)
**Purpose**: Time dimension table for temporal analysis
- **Primary Key**: `Fecha` (date)
- **Total Columns**: 13 columns
- **Key Fields**:
  - `Fecha` - Date (Primary key)
  - `Year` - Year
  - `NumMes` - Month number
  - `Mes` - Month name
  - `Q` - Quarter
  - `Semana` - Week
  - `CALMONTH` - Calendar month (varchar)
  - `YearMes` - Year-Month combination

## Table Relationships

### Primary Relationships

#### **Star Schema Design**
The database follows a **star schema** pattern with `segmentacion` as the central fact table:

```
                    ┌─────────────┐
                    │   tiempo    │
                    │ (Date Dim)  │
                    └─────┬───────┘
                          │
                          │ calday = Fecha
                          │
    ┌─────────────┐      ┌┴─────────────┐      ┌─────────────┐
    │   cliente   │      │ segmentacion │      │  producto   │
    │(Customer Dim)│◄─────┤  (Fact)      ├─────►│(Product Dim)│
    └─────┬───────┘      └┬─────────────┘      └─────────────┘
          │               │
          │ customer_id   │ material_id = Material
          │               │
          │               │
    ┌─────▼───────┐      │
    │cliente_cedi │      │
    │(Bridge Tbl) │      │
    └─────┬───────┘      │
          │               │
          │ cedi_id = CEDIid
          │
    ┌─────▼───────┐
    │   mercado   │
    │(Territory)  │
    └─────────────┘
```

### Detailed Relationship Mapping

#### **1. segmentacion ← → cliente**
- **Relationship**: Many-to-One
- **Join**: `segmentacion.customer_id = cliente.customer_id`
- **Purpose**: Link sales transactions to customer details

#### **2. segmentacion ← → producto**
- **Relationship**: Many-to-One
- **Join**: `segmentacion.material_id = producto.Material`
- **Purpose**: Link sales transactions to product details

#### **3. segmentacion ← → tiempo**
- **Relationship**: Many-to-One
- **Join**: `segmentacion.calday = tiempo.Fecha`
- **Alternative**: `segmentacion.CALMONTH = tiempo.CALMONTH`
- **Purpose**: Enable temporal analysis and time-based aggregations

#### **4. cliente ← → cliente_cedi**
- **Relationship**: One-to-Many
- **Join**: `cliente.customer_id = cliente_cedi.customer_id`
- **Purpose**: Map customers to distribution centers and territories

#### **5. cliente_cedi ← → mercado**
- **Relationship**: Many-to-One
- **Join**: `cliente_cedi.cedi_id = mercado.CEDIid`
- **Purpose**: Get market/territory information for customers

## Common Query Patterns

### 1. **Customer Revenue Analysis**
```sql
SELECT 
    c.Nombre_cliente,
    SUM(s.IngresoNetoSImpuestos) as total_revenue
FROM dev.segmentacion s
INNER JOIN dev.cliente c ON s.customer_id = c.customer_id
GROUP BY c.Nombre_cliente
```

### 2. **Product Performance by Territory**
```sql
SELECT 
    p.Categoria,
    cc.Territorio,
    SUM(s.VentasCajasUnidad) as total_units
FROM dev.segmentacion s
INNER JOIN dev.producto p ON s.material_id = p.Material
INNER JOIN dev.cliente_cedi cc ON s.customer_id = cc.customer_id
GROUP BY p.Categoria, cc.Territorio
```

### 3. **Time-based Sales Analysis**
```sql
SELECT 
    t.Year,
    t.Q as Quarter,
    SUM(s.IngresoNetoSImpuestos) as quarterly_revenue
FROM dev.segmentacion s
INNER JOIN dev.tiempo t ON s.calday = t.Fecha
GROUP BY t.Year, t.Q
ORDER BY t.Year, t.Q
```

## Key Business Metrics

### Revenue Metrics
- **`IngresoNetoSImpuestos`**: Primary revenue metric (net revenue without taxes)
- **`net_revenue`**: Alternative revenue field
- **`VentasCajasUnidad`**: Unit sales (cases sold)
- **`bottles_sold_m`**: Volume sales (bottles sold)

### Dimensional Analysis
- **Customer Segmentation**: By channel, territory, size, industry
- **Product Analysis**: By category, brand, flavor, package type
- **Geographic Analysis**: By region, territory, CEDI, local vs external
- **Temporal Analysis**: By year, quarter, month, week, day

## Data Quality Notes

1. **Schema Prefix**: All tables require `dev.` prefix
2. **Key Fields**: 
   - `customer_id` is the primary customer identifier (NOT cliente_id)
   - `Material` is the primary product identifier
   - `Fecha` is the primary date field
3. **Join Considerations**:
   - Customer ID lengths vary between tables (varchar(56) in cliente, varchar(20) in segmentacion)
   - Material ID referenced as `material_id` in segmentacion, `Material` in producto
4. **Null Handling**: Many fields allow NULL values, consider in WHERE clauses

## Performance Recommendations

1. **Indexing Strategy**:
   - Index on `segmentacion.customer_id`, `segmentacion.material_id`, `segmentacion.calday`
   - Index on `cliente.customer_id`
   - Index on `producto.Material`
   - Index on `tiempo.Fecha`

2. **Query Optimization**:
   - Use date filters on `segmentacion.calday` for time-based queries
   - Prefer INNER JOINs for better performance
   - Use appropriate GROUP BY for aggregations
   - Consider partitioning segmentacion table by date

3. **SQL Server Specific**:
   - Use `TOP` instead of `LIMIT` for result limiting
   - Leverage SQL Server's columnstore indexes for analytical queries
   - Use `WITH (NOLOCK)` for read-only analytical queries if appropriate
