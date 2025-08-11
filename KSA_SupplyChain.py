import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns 
import calendar
from charts.plot_ppie import plot_professional_pie
from charts.plot_pie_newclor import plot_color_pie
from charts.plot_hbar import plot_hbar_chart_
from charts.plot_bar import plot_bar_chart
from charts.plot_line import plot_line_chart
from charts.plot_scatter import plot_scatter_chart
from charts.plot_histogram import plot_histogram_chart
from charts.plot_cat_bar import plot_cbar_chart
from matplotlib.backends.backend_pdf import PdfPages
# General cleaning and EDA
df=pd.read_csv('warehouse_operations_saudi_international.csv')
df=pd.DataFrame(df)
# print(df.info()) 
# print(df.describe())
''' it is the 1st dahboard coding section now '''
fig1,axas=plt.subplots(2,2,figsize=(13,9),facecolor="#F5F7FA")
plt.suptitle("SALES & REVENUE PERFORMANCE DAHBOARD", fontsize=18, fontweight='600',family='Segoe UI',color='#333333')
''' subplot 1 top 10 product cat by revenue'''
df['Total_Cost'] = pd.to_numeric(df['Total_Cost'], errors='coerce')
pro_cat_revenue=((df.groupby(['Product_Category'])['Total_Cost'].sum().sort_values(ascending=False)).head(6)).round(1)
# print(pro_cat_revenue)
plot_cbar_chart(
    pro_cat_revenue.index,
    pro_cat_revenue.values,
    title='TOTAL_REVENUE BY PRODUCT_CATERGORY',
    title_font='Arial',
    xlabel='PRODUCT CATEGORY',
    ylabel='TOTAL REVENUE',
    axis_font='Verdana',
    figure_facecolor="#f5f5f5",
    category_colors=None,
    use_sns_palette=True,
    sns_palette="colorblind",
    rotation=0,
    show_minor_ticks=True,
    show_minor_labels=False,
    grid=True,
    y_format='M',
    ax=axas[0,0])
axas[0,0].tick_params(axis='x', labelsize=7)

'''2nd subplot >>> which region helps to make the most revenue '''
revenue_region=((df.groupby(['Customer_Region'])['Total_Cost'].sum().sort_values(ascending=False)).head(10)).round(1)
# print(revenue_region)
plot_color_pie(revenue_region, title='Top_10 Regions By Revenue ', top_n=10, value_label='SAR',
                   label_font='Verdana', title_font='Arial', show_others=False, start_angle=140,
                   ax=axas[0,1], pie_radius=1.4, pie_center=(0, 0), use_sns_palette=True, sns_palette="YlOrRd",custom_colors=None,legend_format="M",legend_decimals=3)
# plt.show()
''' 3rd subplot montly AVG slaes trend '''
df['Order_Date']=pd.to_datetime(df['Order_Date'])
# df['Year_Month']=df['Order_Date'].dt.to_period('M')
# print(df['Year_Month'])
df['Months'] = df['Order_Date'].dt.month
df['Years'] = df['Order_Date'].dt.year
montly_revenue=df.groupby(['Years','Months'])['Total_Cost'].mean().reset_index()
pivot_data=montly_revenue.pivot(index='Months',columns='Years',values='Total_Cost')
pivot_data=pivot_data.sort_index()
colors = sns.color_palette("bright", n_colors=len(pivot_data.columns))
axas[1,0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1_000:.0f}K'))
axas[1,0].set_xticks(range(1, 13))
axas[1,0].set_xticklabels(calendar.month_abbr[1:])
for i,year in enumerate(pivot_data.columns):
    plot_line_chart(
    pivot_data.index, pivot_data[year].fillna(0).values, 
    title='Average Monthly Revenue Trend by Year', xlabel='Month', ylabel='Total_Revenue(K)', label=str(year), 
    linestyle='solid', linewidth=2, color=colors[i], 
    marker='o', markersize=8, markerfacecolor=None, 
    markeredgecolor='black', legend=True, ax=axas[1,0],
    rotation=0, annotation_freq=None,
    show_rolling_avg=False, rolling_window=3, rolling_color='orange')
'''Subplot no 4 top 10 most revenue generator suppliers'''
suplier_total_reviue=((df.groupby(['Supplier_ID'])['Total_Cost'].sum().sort_values(ascending=False)).head(10)).round(1)
plot_hbar_chart_(
    suplier_total_reviue.index,
    suplier_total_reviue.values,
    title='TOP 10 Suppliers with High Sales',
    title_font='Arial',
    xlabel='Total_Revenue',
    ylabel='Supplier_ID',
    axis_font='Verdana',
    figure_facecolor="#f5f5f5",
    category_colors=None,
    use_sns_palette=True,
    sns_palette="bright",
    show_minor_ticks=True,
    show_minor_labels=False,
    annotate=True,
    grid=True,
    value_format='K',
    extend_xaxis=True,
    ax=axas[1,1])
plt.subplots_adjust(
    left=0.07,    
    right=0.96,   
    top=0.87,     
    bottom=0.1,   
    wspace=0.23,   
    hspace=0.4 
)
# bar positioning 
box = axas[0,0].get_position()  
axas[0,0].set_position([box.x0 + 0.02, box.y0, box.width*1.2, box.height]) 
# pie chart positioning 
box = axas[0,1].get_position()  
axas[0,1].set_position([box.x0 - 0.04, box.y0, box.width, box.height]) 
#horizental bar position 
box = axas[1,1].get_position()  
axas[1,1].set_position([box.x0 - 0.01, box.y0, box.width*1.06, box.height]) 
fig1.text(0.5, 0.01, "KSA Supply Chain: Warehouse Operations Analytics| Analysis by AneesAhmad", 
          ha='center', fontsize=7, style='italic', color='gray')

'''' this is dahboard no 2 for Inventory & Stock Management'''

fig2,axes=plt.subplots(2,2,figsize=(13,9),facecolor="#F5F7FA")
plt.suptitle("Inventory & Stock Management Dashboard", fontsize=18, fontweight='600',family='Segoe UI',color='#333333')
'''subplot1 for Avg reoreder level by product category '''
Avg_reorder_cat=(df.groupby(['Product_Category'])['Reorder_Level'].mean().sort_values(ascending=False)).round(2)
# print(Avg_reorder_cat)
plot_cbar_chart(
    Avg_reorder_cat.index,
    Avg_reorder_cat.values,
    title='Average Reorder Level by Product Category',
    title_font='Arial',
    xlabel='PRODUCT CATEGORY',
    ylabel='Average Reorder Level',
    axis_font='Verdana',
    figure_facecolor="#f5f5f5",
    category_colors=None,
    use_sns_palette=True,
    sns_palette="colorblind",
    rotation=0,
    show_minor_ticks=True,
    show_minor_labels=False,
    grid=True,
    y_format=None,
    ax=axes[0,0])
axes[0,0].grid(True, which='major', axis='y', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axes[0,0].grid(False, which='major', axis='x')
axes[0,0].grid(True, which='minor', axis='y',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')
axes[0,0].tick_params(axis='x', labelsize=7)
'''2nd subplot is for stock level warehouse location & product cateory '''
stock_warhouseloc_procat=(df.groupby(['Warehouse_Location','Product_Category'])['Stock_Level'].sum().unstack())
# print(stock_warhouseloc_procat)
stock_warhouseloc_procat.plot(
    colormap="viridis",
    kind='bar',
    ax=axes[1,1],
    edgecolor='black',
)
axes[1,1].minorticks_on()
axes[1,1].facecolor='#f5f5f5'
axes[1,1].set_title('Total Stock Level of Different Warehouse and Product Category', fontsize=11, fontweight='semibold',family='Arial')
axes[1,1].set_xlabel('Warehouse_Location',fontsize=11,family='Verdana')
axes[1,1].set_ylabel('Total Stock Level',fontsize=11,family='Verdana')
axes[1,1].legend(title='Product_category',fontsize=6,title_fontsize=6,loc='lower right')
axes[1,1].tick_params(axis='x', rotation=0)
axes[1,1].grid(True, which='major', axis='y', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axes[1,1].grid(False, which='major', axis='x')
axes[1,1].grid(True, which='minor', axis='y',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')
axes[1,1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1_000:.0f}K'))
offset = stock_warhouseloc_procat.max().max() * 0.015
# Anotate each bar (loop over container)
for container in axes[1,1].containers:
    for bar in container:
        height = bar.get_height()
        height_k=height/1000
        axes[1,1].text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f'{height_k:.1f}K',
            ha='center',
            va='bottom',
            fontsize=5,
            fontweight='semibold',
            family='Tahoma',
            color=bar.get_facecolor(),
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.1')
        )

''' this is subplot no 3 for top reordering products (most buying )'''
product_reorder=df.groupby(['Product_ID'])['Reorder_Level'].sum().sort_values(ascending=False).head(10)
plot_hbar_chart_(
    product_reorder.index,
    product_reorder.values,
    title='Top 10 Highly In Demand Product & Reorders',
    title_font='Arial',
    xlabel='Order Level',
    ylabel='Product_ID',
    axis_font='Verdana',
    figure_facecolor="#f5f5f5",
    category_colors=None,
    use_sns_palette=True,
    sns_palette="colorblind",
    show_minor_ticks=True,
    show_minor_labels=False,
    annotate=True,
    grid=True,
    value_format=None,
    extend_xaxis=True,
    ax=axes[1,0])
axes[1,0].minorticks_on()
axes[1,0].tick_params(axis='x', rotation=0)
axes[1,0].grid(True, which='major', axis='x', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axes[1,0].grid(False, which='major', axis='y')
axes[1,0].grid(True, which='minor', axis='x',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')


''' this is subplot no 4 for 2nd one abiut total stock distrubtion in different warehouses '''
stock_warehouse=df.groupby(['Warehouse_Location'])['Stock_Level'].sum().sort_values(ascending=False)
plot_color_pie(
    stock_warehouse,
    title='Total Stock Distribution Among Warehouse',
    top_n=6,
    value_label='Quantity',
    label_font='Verdana',
    title_font='Arial',
    show_others=False,
    start_angle=140,
    pie_radius=1.4,
    pie_center=(1, 0),
    use_sns_palette=True,  
    sns_palette="Greens",    
    custom_colors=None,
    legend_format="K",
    ax=axes[0,1]
)

plt.subplots_adjust(
    left=0.07,    
    right=0.95,   
    top=0.87,     
    bottom=0.1,   
    wspace=0.23,   
    hspace=0.4 
)
# positioning of the 1st  bar 
box=axes[0,0].get_position()
axes[0,0].set_position([box.x0+0.01,box.y0,box.width*1.2,box.height])
# positioning of the pie 
box=axes[0,1].get_position()
axes[0,1].set_position([box.x0-0.02,box.y0-0.03,box.width,box.height])
# positioning of the 3rd subplot Hbar
box=axes[1,0].get_position()
axes[1,0].set_position([box.x0+0.01,box.y0,box.width*1.05,box.height])
# positioning of categoies multibar
box=axes[1,1].get_position()
axes[1,1].set_position([box.x0+0.01,box.y0,box.width*1.06,box.height])
fig2.text(0.5, 0.01, "KSA Supply Chain: Warehouse Operations Analytics| Analysis by AneesAhmad", 
          ha='center', fontsize=7, style='italic', color='gray')

''' this is dahboard no 3  Supplier & Procurement Insights Dashboard'''
fig3,axas7=plt.subplots(2,2,figsize=(13,9),facecolor="#F5F7FA")
plt.suptitle("Warehouse & Orders Analysis Dashboard", fontsize=18, fontweight='600',family='Segoe UI',color='#333333')
orders_and_status=df.groupby(['Delivery_Status'])['Quantity'].sum().sort_values(ascending=False)
plot_color_pie(
    orders_and_status,
    title='Total Orders & its Delivery Status',
    top_n=5,
    value_label='Orders',
    label_font='Verdana',
    title_font='Arial',
    show_others=False,
    start_angle=140,
    pie_radius=1.3,
    pie_center=(0, 0),
    use_sns_palette=True,  
    sns_palette="Greens",      
    custom_colors=None,
    ax=axas7[0,1],
    legend_format='K',   
    legend_decimals=2 )

''' this is for warehouseid count in each city & total revnue enerated '''
warhouse_orders=(df.groupby(['Warehouse_ID','Warehouse_Location'])['Quantity'].sum().unstack())
# print(stock_warhouseloc_procat)
warhouse_orders.plot(
    colormap="viridis",
    kind='bar',
    ax=axas7[1,0],
    edgecolor='black',
)
axas7[1,0].minorticks_on()
axas7[1,0].facecolor='#f5f5f5'
axas7[1,0].set_title('Most Active Warhouses in Different Regions', fontsize=11, fontweight='semibold',family='Arial')
axas7[1,0].set_xlabel('Warehouse_ID',fontsize=11,family='Verdana')
axas7[1,0].set_ylabel('Total Orders ',fontsize=11,family='Verdana')
axas7[1,0].legend(title='Warehouse_Location',fontsize=6,title_fontsize=6,loc='lower right')
axas7[1,0].tick_params(axis='x', rotation=0)
axas7[1,0].grid(True, which='major', axis='y', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axas7[1,0].grid(False, which='major', axis='x')
axas7[1,0].grid(True, which='minor', axis='y',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')
axas7[1,0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1_000:.0f}K'))
offset = warhouse_orders.max().max() * 0.015
# Anotate each bar (loop over container)
for container in axas7[1,0].containers:
    for bar in container:
        height = bar.get_height()
        height_k=height/1000
        axas7[1,0].text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f'{height_k:.1f}K',
            ha='center',
            va='bottom',
            fontsize=5,
            fontweight='semibold',
            family='Tahoma',
            color=bar.get_facecolor(),
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.1')
        )
''' this is the 4th subplot for warehouse id location and total revenue '''
warehouse_revenue_loc=(df.groupby(['Warehouse_ID','Warehouse_Location'])['Total_Cost'].sum().unstack())
# print(stock_warhouseloc_procat)
warehouse_revenue_loc.plot(
    colormap="viridis",
    kind='bar',
    ax=axas7[1,1],
    edgecolor='black',
)
axas7[1,1].minorticks_on()
axas7[1,1].facecolor='#f5f5f5'
axas7[1,1].set_title('Total Revenue Generated by Different Warehouses', fontsize=11, fontweight='semibold',family='Arial')
axas7[1,1].set_xlabel('Warehouse_ID',fontsize=11,family='Verdana')
axas7[1,1].set_ylabel('Total Revenue(SAR) ',fontsize=11,family='Verdana')
axas7[1,1].legend(title='Warehouse_Location',fontsize=6,title_fontsize=6,loc='lower right')
axas7[1,1].tick_params(axis='x', rotation=0)
axas7[1,1].grid(True, which='major', axis='y', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axas7[1,1].grid(False, which='major', axis='x')
axas7[1,1].grid(True, which='minor', axis='y',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')
axas7[1,1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1_000_000:.0f}M'))
offset = warehouse_revenue_loc.max().max() * 0.015
# Anotate each bar (loop over container)
for container in axas7[1,1].containers:
    for bar in container:
        height = bar.get_height()
        height_k=height/1000000
        axas7[1,1].text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f'{height_k:.1f}M',
            ha='center',
            va='bottom',
            fontsize=5,
            fontweight='semibold',
            family='Tahoma',
            color=bar.get_facecolor(),
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.1')
        )

''' this is heat map for my final subplot '''
QUantity_priority_mode=df.groupby(['Shipping_Mode','Priority'])["Quantity"].sum().reset_index()
# print(Avg_time_m_per_km)
heatmap_data=QUantity_priority_mode.pivot(
    index='Shipping_Mode',
    columns='Priority',
    values='Quantity'
    
)
annot_k = heatmap_data.applymap(lambda v: f"{v/1000:.1f}K" if v>1000 else f"{v:.0f}")
sns.heatmap(
    heatmap_data,
    annot=annot_k,
    fmt='',
    cmap='YlGnBu',
    linewidths=0.5,
    linecolor='gray',
    ax=axas7[0,0],
    cbar_kws={'label':'Total Orders'}
)
axas7[0,0].set_title("Order Priority and Shipping Mode", fontsize=12, fontweight='semibold', pad=10,family='Arial')
axas7[0,0].set_xlabel("Priority", fontsize=11, fontweight='normal',family='Verdana')
axas7[0,0].set_ylabel("Shipping Mode", fontsize=11, fontweight='normal',family='Verdana')
axas7[0,0].tick_params(axis='x', rotation=0)
axas7[0,0].tick_params(axis='y', rotation=0)
plt.subplots_adjust(
    left=0.064, 
    bottom=0.073,   
    right=0.986,   
    top=0.888,     
    wspace=0.162,   
    hspace=0.289
)
box=axas7[0,1].get_position()
axas7[0,1].set_position([box.x0-0.06,box.y0-0.02,box.width,box.height])
fig3.text(0.5, 0.01, "KSA Supply Chain: Warehouse Operations Analytics| Analysis by AneesAhmad", 
          ha='center', fontsize=7, style='italic', color='gray')

''' this is dashboard no 4 for  SUppliers analysis   showing '''
fig4,axis=plt.subplots(2,2,figsize=(13,9),facecolor="#F5F7FA")
plt.suptitle("Supplier & Procurement Insights Dashboard", fontsize=18, fontweight='600',family='Segoe UI',color='#333333')
supplierco_revenue=(df.groupby(['Supplier_Name'])['Total_Cost'].sum().sort_values(ascending=False)).round(1)
# print(supplierco_revenue)
plot_cbar_chart(
    supplierco_revenue.index,
    supplierco_revenue.values,
    title='Supplier Companies & Total Revenue',
    title_font='Arial',
    xlabel='Company Name',
    ylabel='Total Sales(SAR)',
    axis_font='Verdana',
    figure_facecolor="#f5f5f5",
    category_colors=None,
    use_sns_palette=True,
    sns_palette="rocket_r",
    rotation=0,
    show_minor_ticks=True,
    show_minor_labels=False,
    grid=True, 
    y_format='M',  
    ax=axis[0,0]
)
axis[0,0].minorticks_on()
axis[0,0].tick_params(axis='x', rotation=0,labelsize=9)
axis[0,0].grid(True, which='major', axis='y', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axis[0,0].grid(False, which='major', axis='x')
axis[0,0].grid(True, which='minor', axis='y',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')

''' this is subplot no 2 order distribution among the suppliers '''
order_suppliers=df.groupby(["Supplier_Name"])["Quantity"].sum().sort_values(ascending=False)
plot_color_pie(order_suppliers, title='Sales Distribution Among Supplier', top_n=10, value_label='Orders',
                   label_font='Verdana', title_font='Arial', show_others=False, start_angle=140,
                   ax=axis[0,1], pie_radius=1.4, pie_center=(0, 0), use_sns_palette=True, sns_palette="Paired",legend_format="K",custom_colors=None)
'''this is subplot 3 for AVg delay days by differetnt suppliers comapny '''
delay_suppliers=(df.groupby(['Supplier_Name'])["Delay_Days"].mean().sort_values(ascending=False)).round(1)
plot_cbar_chart(
    delay_suppliers.index,
    delay_suppliers.values,
    title='Supplier Companies & Average Delay(DAYS)',
    title_font='Arial',
    xlabel='Company Name',
    ylabel='Time Delay(Days)',
    axis_font='Verdana',
    figure_facecolor="#f5f5f5",
    category_colors=None,
    use_sns_palette=True,
    sns_palette="rocket_r",
    rotation=0,
    show_minor_ticks=True,
    show_minor_labels=False,
    grid=True, 
    y_format=None,  
    ax=axis[1,0]
)
axis[1,0].minorticks_on()
axis[1,0].tick_params(axis='x', rotation=0,labelsize=8)
axis[1,0].grid(True, which='major', axis='y', linestyle="--",alpha=0.7, linewidth=0.7, zorder=0,color='black')
axis[1,0].grid(False, which='major', axis='x')
axis[1,0].grid(True, which='minor', axis='y',alpha=0.4, linestyle=":", linewidth=0.4, zorder=0,color='black')
""" this is subplot no 4 for deliverd suppliers and its mode of dilivery """
metric = "Quantity"
deliv_flag = df["Delivery_Status"].astype(str).str.lower().eq("delivered")
totals = (df.groupby(["Supplier_Name","Shipping_Mode"])[metric]
            .sum().rename("total"))
delivered = (df[deliv_flag].groupby(["Supplier_Name","Shipping_Mode"])[metric]
               .sum().rename("delivered"))
rate = (pd.concat([totals, delivered], axis=1)
          .fillna(0.0))
rate["delivered_rate"] = np.where(rate["total"]>0, rate["delivered"]/rate["total"], 0.0)
rate = rate.reset_index()
topN = 5
top_suppliers = (totals.groupby("Supplier_Name").sum()
                 .sort_values(ascending=False).head(topN).index)
rate = rate[rate["Supplier_Name"].isin(top_suppliers)]

pivot_rate = (rate.pivot(index="Supplier_Name", columns="Shipping_Mode", values="delivered_rate")
                  .fillna(0.0)
                  [sorted(rate["Shipping_Mode"].unique())])  

pivot_rate.plot(
    kind="bar",
    ax=axis[1,1],
    colormap="viridis",   
    edgecolor="black",
)

axis[1,1].minorticks_on()
axis[1,1].facecolor = "#f5f5f5"
axis[1,1].set_title("Delivered % by Supplier and Shipping Mode", fontsize=11, fontweight="semibold", family="Arial")
axis[1,1].set_xlabel("Company Name", fontsize=11, family="Verdana")
axis[1,1].set_ylabel("Delivered Percentage", fontsize=11, family="Verdana")
axis[1,1].tick_params(axis="x", rotation=0,labelsize=8)
axis[1,1].grid(True, which="major", axis="y", linestyle="--", alpha=0.7, linewidth=0.7, zorder=0, color="black")
axis[1,1].grid(False, which="major", axis="x")
axis[1,1].grid(True, which="minor", axis="y", alpha=0.4, linestyle=":", linewidth=0.4, zorder=0, color="black")
axis[1,1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=0)) 
axis[1,1].set_ylim(0,0.3)
axis[1,1].legend(title="Shipping_Mode", fontsize=6, title_fontsize=6, loc="lower right")

offset = 0.001  
for container in axis[1,1].containers:
    for bar in container:
        h = bar.get_height()
        if np.isnan(h):  
            continue
        axis[1,1].text(
            bar.get_x() + bar.get_width()/2,
            h + offset,
            f"{h:.0%}",
            ha="center", va="bottom",
            fontsize=6, fontweight="semibold", family="Tahoma",
            color=bar.get_facecolor(),
            bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.1")
        )

plt.subplots_adjust(
    left=0.07,    
    right=0.95,   
    top=0.87,     
    bottom=0.1,   
    wspace=0.23,   
    hspace=0.4 
)
# positioning of the bar subplot 1
box=axis[0,0].get_position()
axis[0,0].set_position([box.x0+0.003,box.y0,box.width,box.height])
 # positioning of the piee subplot 2
box=axis[0,1].get_position()
axis[0,1].set_position([box.x0-0.04,box.y0+0.002,box.width,box.height])
# positioning of the 3rd subplot Hbar
box=axis[1,0].get_position()
axis[1,0].set_position([box.x0,box.y0,box.width,box.height])
# positioning of categoies multibar
box=axis[1,1].get_position()
axis[1,1].set_position([box.x0+0.010,box.y0,box.width*1.06,box.height])
fig4.text(0.5, 0.01, "KSA Supply Chain: Warehouse Operations Analytics| Analysis by AneesAhmad", 
          ha='center', fontsize=7, style='italic', color='gray')

'''' thsi is dahboar no 5 for monthly stocks trends'''
fig5,axos=plt.subplots(2,3,figsize=(13,9),facecolor="#F5F7FA")
plt.suptitle("Monthly Stocks Trend & Product Category", fontsize=18, fontweight='600',family='Segoe UI',color='#333333')
df['Order_Date']=pd.to_datetime(df['Order_Date'])
# df['Year_Month']=df['Order_Date'].dt.to_period('M')
# print(df['Year_Month'])
df['Months'] = df['Order_Date'].dt.month
df['Years'] = df['Order_Date'].dt.year
montly_revenue=df.groupby(['Product_Category','Months'])['Stock_Level'].mean().reset_index()
pivot_data=montly_revenue.pivot(index='Months',columns='Product_Category',values='Stock_Level')
pivot_data=pivot_data.sort_index()
colors = sns.color_palette("Paired", n_colors=len(pivot_data.columns))
for i,year in enumerate([2021,2022,2023,2024,2025]):
    yeardata=df[df["Years"]==year]
    monthly_stock_year=yeardata.groupby(['Product_Category','Months'])['Stock_Level'].mean().reset_index()
    pivot_year_data=monthly_stock_year.pivot(index='Months',columns='Product_Category',values='Stock_Level')
    row,col=divmod(i,3)
    aix=axos[row,col]
    aix.set_xticks(range(1, 13))
    aix.set_xticklabels(calendar.month_abbr[1:])

    for j,category in enumerate(pivot_year_data.columns):
        plot_line_chart(
        pivot_year_data.index, pivot_year_data[category].fillna(0).values, 
        label=category, 
        linestyle='solid', linewidth=1, color=colors[j],     
        marker='o', markersize=7, markerfacecolor=None, 
        markeredgecolor='black', legend=False,
        rotation=0, annotation_freq=None,ax=aix,
        show_rolling_avg=False, rolling_window=3, rolling_color='orange')
    aix.set_title(f'Stock Trend in Year {year}', fontsize=11, fontweight='semibold', family='Arial')
    aix.set_xlabel('Month', fontsize=10, family='Verdana')
    aix.set_ylabel('Stock Level', fontsize=10, family='Verdana')
    aix.legend(title='Product Category', fontsize=4, title_fontsize=5, loc='upper right',framealpha=0.)
    aix.grid(True, which='major', axis='y', linestyle="--", alpha=0.7, linewidth=0.7, zorder=0, color='black')
    aix.grid(False, which='major', axis='x')
    aix.grid(True, which='minor', axis='y', alpha=0.4, linestyle=":",linewidth=0.4,zorder=0,color='black')
''' this is subplot no 6 for overall general overview (avg stock trend in thsese years )'''
yearly_stock_trend=(df.groupby(['Months','Product_Category'])['Stock_Level'].mean()).round(1).reset_index()
pivot_overall=yearly_stock_trend.pivot(index='Months',columns='Product_Category',values='Stock_Level')
aiix = axos[1, 2] 
colors = sns.color_palette("Paired", n_colors=len(pivot_overall.columns))
for j, category in enumerate(pivot_overall.columns):
     plot_line_chart(
        pivot_overall.index, pivot_overall[category].fillna(0).values, 
        label=category, 
        linestyle='solid', linewidth=1, color=colors[j],     
        marker='o', markersize=7, markerfacecolor=None,
        markeredgecolor='black', legend=False,
        rotation=0, annotation_freq=None,ax=aiix,
        show_rolling_avg=False, rolling_window=3, rolling_color='orange')
     aiix.set_title('Overall Stock Trend (2021-2025)', fontsize=11, fontweight='semibold', family='Arial')
     aiix.set_xlabel('Month', fontsize=10, family='Verdana')
     aiix.set_ylabel('Stock Level', fontsize=10, family='Verdana')
     aiix.legend(title='Product Category', fontsize=4, title_fontsize=5, loc='upper right',framealpha=0.)
     aiix.grid(True, which='major', axis='y', linestyle="--", alpha=0.7, linewidth=0.7, zorder=0, color='black')
     aiix.grid(False, which='major', axis='x')
     aiix.grid(True, which='minor', axis='y', alpha=0.4, linestyle=":",linewidth=0.4,zorder=0,color='black')
     aiix.set_xticks(range(1, 13))
     aiix.set_xticklabels(calendar.month_abbr[1:])

plt.subplots_adjust(
    left=0.056, 
    bottom=0.061,   
    right=0.985,   
    top=0.9,     
    wspace=0.2,   
    hspace=0.302
)
fig5.text(0.5, 0.01, "KSA Supply Chain: Warehouse Operations Analytics| Analysis by AneesAhmad", 
          ha='center', fontsize=7, style='italic', color='gray')
''' this is dashboard no 5 for montlhy sales trend and product category for 4 years '''
fig6,axus=plt.subplots(2,3,figsize=(13,9),facecolor="#F5F7FA")
plt.suptitle("Average Monthly Sales Trend & Product Category", fontsize=18, fontweight='600',family='Segoe UI',color='#333333')
df['Order_Date']=pd.to_datetime(df['Order_Date'])
# df['Year_Month']=df['Order_Date'].dt.to_period('M')
# print(df['Year_Month'])
df['Months'] = df['Order_Date'].dt.month
df['Years'] = df['Order_Date'].dt.year
montly_sales=df.groupby(['Product_Category','Months'])['Quantity'].mean().reset_index()
pivot_data_sales=montly_sales.pivot(index='Months',columns='Product_Category',values='Quantity')
pivot_data_sales=pivot_data_sales.sort_index()
colors = sns.color_palette("Paired", n_colors=len(pivot_data_sales.columns))
for i,year in enumerate([2021,2022,2023,2024,2025]):
    yeardata=df[df["Years"]==year]
    monthly_Sales_year=yeardata.groupby(['Product_Category','Months'])['Quantity'].mean().reset_index()
    pivot_year_Sales_data=monthly_Sales_year.pivot(index='Months',columns='Product_Category',values='Quantity')
    row,col=divmod(i,3)
    aex=axus[row,col]
    aex.set_xticks(range(1, 13))
    aex.set_xticklabels(calendar.month_abbr[1:])

    for k,category in enumerate(pivot_year_Sales_data.columns):
        plot_line_chart(
        pivot_year_Sales_data.index, pivot_year_Sales_data[category].fillna(0).values, 
        label=category, 
        linestyle='solid', linewidth=1, color=colors[k],     
        marker='o', markersize=7, markerfacecolor=None, 
        markeredgecolor='black', legend=False,
        rotation=0, annotation_freq=None,ax=aex,
        show_rolling_avg=False, rolling_window=3, rolling_color='orange')
    aex.set_title(f'Sales Trend in Year {year}', fontsize=11, fontweight='semibold', family='Arial')
    aex.set_xlabel('Month', fontsize=10, family='Verdana')
    aex.set_ylabel('Quantity Sold', fontsize=10, family='Verdana')
    aex.legend(title='Product Category', fontsize=4, title_fontsize=5, loc='upper right',framealpha=0.)
    aex.grid(True, which='major', axis='y', linestyle="--", alpha=0.7, linewidth=0.7, zorder=0, color='black')
    aex.grid(False, which='major', axis='x')
    aex.grid(True, which='minor', axis='y', alpha=0.4, linestyle=":",linewidth=0.4,zorder=0,color='black')
''' this is subplot no 6 for overall general overview (avg stock trend in thsese years )'''
yearly_Sales_trend=(df.groupby(['Months','Product_Category'])['Quantity'].mean()).round(1).reset_index()
pivot_overall_Sales=yearly_Sales_trend.pivot(index='Months',columns='Product_Category',values='Quantity')
aux=axus[1,2] 
colors = sns.color_palette("Paired", n_colors=len(pivot_overall.columns))
for j, category in enumerate(pivot_overall_Sales.columns):
     plot_line_chart(
        pivot_overall_Sales.index, pivot_overall_Sales[category].fillna(0).values, 
        label=category, 
        linestyle='solid', linewidth=1, color=colors[j],     
        marker='o', markersize=7, markerfacecolor=None,
        markeredgecolor='black', legend=False,
        rotation=0, annotation_freq=None,ax=aux,
        show_rolling_avg=False, rolling_window=3, rolling_color='orange')
     aux.set_title('Overall Sales Trend (2021-2025)', fontsize=11, fontweight='semibold', family='Arial')
     aux.set_xlabel('Month', fontsize=10, family='Verdana')
     aux.set_ylabel('Quantity Sold', fontsize=10, family='Verdana')
     aux.legend(title='Product Category', fontsize=4, title_fontsize=5, loc='upper right',framealpha=0.)
     aux.grid(True, which='major', axis='y', linestyle="--", alpha=0.7, linewidth=0.7, zorder=0, color='black')
     aux.grid(False, which='major', axis='x')
     aux.grid(True, which='minor', axis='y', alpha=0.4, linestyle=":",linewidth=0.4,zorder=0,color='black')
     aux.set_xticks(range(1, 13))
     aux.set_xticklabels(calendar.month_abbr[1:])

plt.subplots_adjust(
    left=0.053, 
    bottom=0.062,   
    right=0.983,   
    top=0.9,     
    wspace=0.191,   
    hspace=0.296
)
fig6.text(0.5, 0.01, "KSA Supply Chain: Warehouse Operations Analytics| Analysis by AneesAhmad", 
          ha='center', fontsize=7, style='italic', color='gray')
''' this is dashboard no 6 for total reveue on monthly basis for differnt catgory '''
fig7,axos=plt.subplots(2,3,figsize=(13,9),facecolor="#F5F7FA")
plt.suptitle("Average Monthly Revenue(SAR) Trend & Product Category", fontsize=18, fontweight='600',family='Segoe UI',color='#333333')
df['Order_Date']=pd.to_datetime(df['Order_Date'])
# df['Year_Month']=df['Order_Date'].dt.to_period('M')
# print(df['Year_Month'])
df['Months'] = df['Order_Date'].dt.month
df['Years'] = df['Order_Date'].dt.year
Avg_monthly_revenue=(df.groupby(['Product_Category','Months'])['Total_Cost'].mean().reset_index()).round(1)
pivot_data_revenue=Avg_monthly_revenue.pivot(index='Months',columns='Product_Category',values='Total_Cost')
pivot_data_revenue=pivot_data_revenue.sort_index()
colors = sns.color_palette("Paired", n_colors=len(pivot_data_revenue.columns))
for L,years in enumerate([2021,2022,2023,2024,2025]):
    yeardata=df[df["Years"]==years]
    monthly_revenue_year=yeardata.groupby(['Product_Category','Months'])['Total_Cost'].mean().reset_index()
    pivot_year_revenue_data=monthly_revenue_year.pivot(index='Months',columns='Product_Category',values='Total_Cost')
    row,col=divmod(L,3)
    aeex=axos[row,col]
    aeex.set_xticks(range(1, 13))
    aeex.set_xticklabels(calendar.month_abbr[1:])

    for M,category in enumerate(pivot_year_revenue_data.columns):
        plot_line_chart(
        pivot_year_revenue_data.index, pivot_year_revenue_data[category].fillna(0).values, 
        label=category, 
        linestyle='solid', linewidth=1, color=colors[M],     
        marker='o', markersize=7, markerfacecolor=None, 
        markeredgecolor='black', legend=False,
        rotation=0, annotation_freq=None,ax=aeex,
        show_rolling_avg=False, rolling_window=3, rolling_color='orange')
    aeex.set_title(f'Average Revenue Trend in Year {year}', fontsize=11, fontweight='semibold', family='Arial')
    aeex.set_xlabel('Month', fontsize=10, family='Verdana')
    aeex.set_ylabel('Revenue Generated', fontsize=10, family='Verdana')
    aeex.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1_000:.0f}K'))
    aeex.legend(title='Product Category', fontsize=4, title_fontsize=5, loc='upper right',framealpha=0.6)
    aeex.grid(True, which='major', axis='y', linestyle="--", alpha=0.7, linewidth=0.7, zorder=0, color='black')
    aeex.grid(False, which='major', axis='x')
    aeex.grid(True, which='minor', axis='y', alpha=0.4, linestyle=":",linewidth=0.4,zorder=0,color='black')
''' this is subplot no 6 for overall general overview (avg stock trend in thsese years )'''
yearly_revenue_trend=(df.groupby(['Months','Product_Category'])['Total_Cost'].mean()).round(1).reset_index()
pivot_overall_revenue=yearly_revenue_trend.pivot(index='Months',columns='Product_Category',values='Total_Cost')
auux=axos[1,2] 
colors = sns.color_palette("Paired", n_colors=len(pivot_overall_revenue.columns))
for N, category in enumerate(pivot_overall_revenue.columns):
     plot_line_chart(
        pivot_overall_revenue.index, pivot_overall_revenue[category].fillna(0).values, 
        label=category, 
        linestyle='solid', linewidth=1, color=colors[N],     
        marker='o', markersize=7, markerfacecolor=None,
        markeredgecolor='black', legend=False,
        rotation=0, annotation_freq=None,ax=auux,
        show_rolling_avg=False, rolling_window=3, rolling_color='orange')
     auux.set_title('Overall Revenue Trend (2021-2025)', fontsize=11, fontweight='semibold', family='Arial')
     auux.set_xlabel('Month', fontsize=10, family='Verdana')
     auux.set_ylabel('Revenue Genrated', fontsize=10, family='Verdana')
     auux.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1_000:.0f}K'))
     auux.legend(title='Product Category', fontsize=4, title_fontsize=5, loc='upper right',framealpha=0.)
     auux.grid(True, which='major', axis='y', linestyle="--", alpha=0.7, linewidth=0.7, zorder=0, color='black')
     auux.grid(False, which='major', axis='x')
     auux.grid(True, which='minor', axis='y', alpha=0.4, linestyle=":",linewidth=0.4,zorder=0,color='black')
     auux.set_xticks(range(1, 13))
     auux.set_xticklabels(calendar.month_abbr[1:])

plt.subplots_adjust(
    left=0.053, 
    bottom=0.062,   
    right=0.983,   
    top=0.9,     
    wspace=0.191,   
    hspace=0.296
)
fig7.text(0.5, 0.01, "KSA Supply Chain: Warehouse Operations Analytics| Analysis by AneesAhmad", 
          ha='center', fontsize=7, style='italic', color='gray')
# with PdfPages('KSA_SUPPLYCHAIN_DAHBOARD.pdf') as pdf:
#     pdf.savefig(fig1)
#     pdf.savefig(fig2)
#     pdf.savefig(fig3)
#     pdf.savefig(fig4)
#     pdf.savefig(fig5)
#     pdf.savefig(fig6)
#     pdf.savefig(fig7)
plt.show()