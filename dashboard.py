import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash.dependencies
import numpy as np
import io
import base64
import seaborn as sns
import matplotlib.pyplot as plt

# Sample dataframe
df = pd.read_csv('Jan-24 project.csv')
# Initialize the Dash app
app = dash.Dash(__name__)

# Define categories
categories = df['category'].unique()

# Define the layout of the dashboard
app.layout = html.Div([
    # Wrap the dropdown in a fixed-positioned div
    html.Div([
        dcc.Dropdown(
            id='category-dropdown',
            options=[{'label': category, 'value': category} for category in categories],
            value=categories[0]
        ),
    ], style={'position': 'fixed', 'top': 0, 'left': 0, 'zIndex': 999,'width': '300px'}),
    
    # Main content area
    html.Div([
        dcc.Graph(id='count-plot'),
        dcc.Graph(id='box-plot'),
        html.Img(id='kde-image'),
        # dcc.Graph(id='strip-plot'),
        dcc.Graph(id='sales-by-count-pie-plot'),
        dcc.Graph(id='sales-by-volume-pie-plot'),
        html.Div([
            dcc.Dropdown(
                id='subcategory-dropdown-container',
                multi=True,
                placeholder="Select subcategories"
            ),
            # dcc.Graph(id='average-review-count-plot-by-brand'),
        ]),
        # dcc.Graph(id='average-ratings-plot-by-brand'),
        dcc.Graph(id='review-count-vs-rating-scatter-plot'),
        dcc.Graph(id='sales-vs-price-brackets-stacked-bar-plot'),
        dcc.Graph(id='discount-sale-plot'),
        html.Img(id='good_rating-image'),
        dcc.Graph(id='bubble-plot'),
        # dcc.Graph(id='scatter-plot'),
        
        
    ], style={'margin-top': '50px'})  # Adjust margin to prevent content from being overlapped by the fixed dropdown
])



# Define callback to update the count plot based on category selection
@app.callback(
    Output('count-plot', 'figure'),
    [Input('category-dropdown', 'value')]
)
def update_count_plot(selected_category):
    df_filtered = df[df['category'] == selected_category]
    fig = px.histogram(df_filtered, x='product_subcategory', title=f'Product Subcategory Distribution for {selected_category}')
    fig.update_xaxes(title='Product Subcategory')
    fig.update_yaxes(title='Count')
    return fig



# Define callback to update the average review count plot based on category selection
@app.callback(
    Output('sales-by-count-pie-plot', 'figure'),
    [Input('category-dropdown', 'value')]
)
def update_average_review_count_plot(selected_category):
    df_filtered = df[df['category'] == selected_category]
    average_review_count = df_filtered.groupby('product_subcategory')['customer_reviews_count'].sum().reset_index()
    
    fig = px.pie(average_review_count, values='customer_reviews_count', names='product_subcategory', title=f'Sales by Count for {selected_category}')
    fig.update_xaxes(title='Product Subcategory', tickangle=45)
    fig.update_yaxes(title='Review Count')
    return fig


@app.callback(
    Output('sales-by-volume-pie-plot', 'figure'),
    [Input('category-dropdown', 'value')]
)
def update_average_review_count_plot(selected_category):
    df_filtered = df[df['category'] == selected_category]
    # find amount of sales by multiplying selling price and customer reviews count
    df_filtered['sales'] = df_filtered['selling_price'] * df_filtered['customer_reviews_count']
    average_sale_by_volume = df_filtered.groupby('product_subcategory')['sales'].sum().reset_index()

    fig = px.pie(average_sale_by_volume, values='sales', names='product_subcategory', title=f'Sales by Volume for {selected_category}')
    return fig


@app.callback(
    Output('subcategory-dropdown-container', 'options'),
    [Input('category-dropdown', 'value')]
)
def update_subcategory_dropdown(selected_category):
    df_filtered = df[df['category'] == selected_category]
    subcategories = df_filtered['product_subcategory'].unique()
    return subcategories





# new
@app.callback(
    Output('review-count-vs-rating-scatter-plot', 'figure'),
    [Input('category-dropdown', 'value'),
     Input('subcategory-dropdown-container', 'value')]
)
def update_average_review_count_plot_by_brand(selected_category, selected_subcategories):
    if selected_subcategories is None:
        df_filtered = df[df['category'] == selected_category]
        selected_subcategories = ["All brands"]
    else:
        df_filtered = df[(df['category'] == selected_category) & df['product_subcategory'].isin(selected_subcategories)]
    #drop rows where product rating or customer reviews count is null
    df_filtered = df_filtered.dropna(subset=['product_rating', 'customer_reviews_count'])
    
    df_filtered["weighted_rating"] = df_filtered["product_rating"] * df_filtered["customer_reviews_count"]

    brand_ratings = df_filtered.groupby('brand').agg(total_weighted_rating=('weighted_rating', 'sum'),
                                        total_reviews=('customer_reviews_count', 'sum'))
    brand_ratings['weighted_avg_rating'] = brand_ratings['total_weighted_rating'] / brand_ratings['total_reviews']
    brand_ratings.reset_index(inplace=True)
    fig = px.scatter(brand_ratings, x='weighted_avg_rating', y='total_reviews')
    
    fig.update_xaxes(title='Weighted Avg Rating')
    fig.update_yaxes(title='Review Count(sales)')
    return fig


#where to focus most
@app.callback(
    Output('bubble-plot', 'figure'),
    [Input('category-dropdown', 'value'),
     Input('subcategory-dropdown-container', 'value')]
)
def update_average_review_count_plot_by_brand(selected_category, selected_subcategories):
    
    df_filtered = df[df['category'] == selected_category]
    group_by_column = 'product_type'

    #drop rows where product rating or customer reviews count is null
    df_filtered = df_filtered.dropna(subset=['product_rating', 'customer_reviews_count'])
    
    df_filtered["weighted_rating"] = df_filtered["product_rating"] * df_filtered["customer_reviews_count"]

    brand_ratings = df_filtered.groupby(group_by_column).agg(total_weighted_rating=('weighted_rating', 'sum'),
                                        total_reviews=('customer_reviews_count', 'sum'),avg_selling_price = ('selling_price','mean'))
    brand_ratings['weighted_avg_rating'] = brand_ratings['total_weighted_rating'] / brand_ratings['total_reviews']
    brand_ratings.reset_index(inplace=True)
    fig = px.scatter(brand_ratings, x=group_by_column, y='weighted_avg_rating',size="avg_selling_price",color=group_by_column)
    
    fig.update_xaxes(title= group_by_column)
    fig.update_yaxes(title='Weighted Avg Rating',range=[0,5])
    return fig


# does promotion matter?
@app.callback(
    Output('sales-vs-price-brackets-stacked-bar-plot', 'figure'),
    [Input('category-dropdown', 'value'),
     Input('subcategory-dropdown-container', 'value')]
)
def update_average_review_count_plot_by_brand(selected_category, selected_subcategories):
    if selected_subcategories is None:
        df_filtered = df[df['category'] == selected_category]
        selected_subcategories = ["All brands"]
    else:
        df_filtered = df[(df['category'] == selected_category) & df['product_subcategory'].isin(selected_subcategories)]
    
    quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1]
    price_brackets = df_filtered['selling_price'].quantile(quantiles)
    
    # Create 5 price brackets of selling price and plot a stacked bar chart with hue as demographic
    df_filtered['price_bracket'] = pd.cut(df_filtered['selling_price'], bins=price_brackets, labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])


    # Step 3: Calculate total sales within each price bracket
    sales_by_price_bracket = df_filtered.groupby(['price_bracket','promotion_indicator'])['customer_reviews_count'].mean().reset_index()
    fig = px.bar(sales_by_price_bracket, x='price_bracket', y='customer_reviews_count',color='promotion_indicator', barmode='group',
             title='Sales by Price Bracket',
             labels={'price_bracket': 'Price Bracket', 'customer_reviews_count': 'Total Sales(avg review count)'})
    
    return fig



# heatmap
@app.callback(
    Output('discount-sale-plot', 'figure'),
    [Input('category-dropdown', 'value'),
     Input('subcategory-dropdown-container', 'value')]
)
def update_average_review_count_plot_by_brand(selected_category, selected_subcategories):
    if selected_subcategories is None:
        df_filtered = df[df['category'] == selected_category]
        selected_subcategories = ["All brands"]
    else:
        df_filtered = df[(df['category'] == selected_category) & df['product_subcategory'].isin(selected_subcategories)]
    

    # find discount percentage from selling price and original price
    df_filtered['discount_percentage'] = ((df_filtered['original_price'] - df_filtered['selling_price']) / df_filtered['original_price']) * 100
    # plot discount sales scatter plot
    fig = px.scatter(df_filtered, x='discount_percentage', y='customer_reviews_count',trendline="ols",title='Discount vs Sales')
    return fig


# Define callback to update the kde plot image based on category selection
@app.callback(
    Output('kde-image', 'src'),
    [Input('category-dropdown', 'value')]
)
def update_kde_image(selected_category):
    df_filtered = df[df['category'] == selected_category]
    plt.figure(figsize=(20, 10), dpi=80)
    sns.displot(df_filtered, x="product_rating", hue="product_subcategory", kind="kde", fill=True,warn_singular=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # plt.close()
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    return "data:image/png;base64,{}".format(data)



# Define callback to update the scatter plot based on category selection
@app.callback(
    Output('good_rating-image', 'src'),
    [Input('category-dropdown', 'value')]
)
def update_good_rating_plot(selected_category):
    df_filtered = df[df['category'] == selected_category]

    products_count_per_product_subcategory = df_filtered['product_subcategory'].value_counts()
    # Total products with rating > 4 in each category
    products_with_more_than_4_reviews = df_filtered[df_filtered['product_rating'] > 4]['product_subcategory'].value_counts()
    # Calculate percentage
    percentage_per_product_subcategory = (products_with_more_than_4_reviews / products_count_per_product_subcategory) * 100
    plt.figure(figsize=(14, 7))
    sns.barplot(x=percentage_per_product_subcategory.index, y=percentage_per_product_subcategory.values)
    plt.xlabel('Category')
    plt.ylabel('Percentage')
    plt.title('Percentage of Products with Rating > 4 in Each Sub-Category')
    plt.xticks(rotation=20)
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    # plt.close()
    data = base64.b64encode(buf1.getvalue()).decode("utf-8")
    buf1.close()
    return "data:image/png;base64,{}".format(data)
    

@app.callback(
    Output('box-plot', 'figure'),
    [Input('category-dropdown', 'value')]
)
def update_box_plot(category):
    # Filter the dataframe based on the selected category
    filtered_df = df[df['category'] == category]

    # Create the Plotly figure
    fig = px.box(filtered_df, x='product_subcategory', y='selling_price', title=f"Box Plot of Selling Price for {category}")

    return fig


    

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
