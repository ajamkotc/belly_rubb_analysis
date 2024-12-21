# Data Dictionary

## Dataset: Orders

### Dataset Overview
- **Source**: Square POS system
- **Collected On**: December 20, 2024
- **Description**: This dataset contains individual order data for Belly Rubb.

### Field Definitions

| Column Name              | Description                                                         | Type     | Units                   | Example Value                            | Notes                                      |
| ------------------------ | ------------------------------------------------------------------- | -------- | ----------------------- | ---------------------------------------- | ------------------------------------------ |
| Order                    | Platform ordered from with order number                             | String   | N/A                     | 'Uber Eats Delivery 8F819'               |                                            |
| Order Date               | Date and time ordered                                               | Datetime | yyyy-mm-dd              | 2023-12-18                               |                                            |
| Currency                 | Currency used                                                       | String   | N/A                     | 'USD'                                    | Constant                                   |
| Order Subtotal           | Subtotal of order                                                   | Float    | USD                     | 19.99                                    |                                            |
| Order Shipping Price     | Shipping price of the order                                         |          |                         |                                          | All missing                                |
| Order Tax Total          | Tax amount applied to subtotal                                      | Float    | USD                     | 3.46                                     |                                            |
| Order Total              | Order amount including subtotal and tax                             | Float    | USD                     | 112.47                                   |                                            |
| Order Refunded Amount    | Amount refunded                                                     |          |                         |                                          | All missing                                |
| Fulfillment Date         | Date order was fulfilled                                            | Datetime | mm/dd/yyyy, hh:mm PM/AM | 12/18/2024, 1:58 PM                      |                                            |
| Fulfillment Type         | How order was fulfilled                                             | String   | N/A                     | 'Pickup'                                 |                                            |
| Fulfillment Status       | Whether order was fulfilled or not                                  | String   | N/A                     | 'Completed'                              | Constant                                   |
| Channels                 | Channels with which ordered                                         | String   | N/A                     | 'DoorDash'                               |                                            |
| Fulfillment Location     | Location which fulfilled order                                      | String   | N/A                     | 'Belly Rubb'                             | Constant                                   |
| Fulfillment Notes        | Special order requests                                              | String   | N/A                     | 'Forks please'                           | 89.7% missing                              |
| Recipient Name           | Name of customer                                                    | String   | N/A                     | 'Diana O.'                               |                                            |
| Recipient Email          | Email of customer or POS integration                                | String   | N/A                     | 'point-of-sale-integration@doordash.com' | 23.8% missing                              |
| Recipient Phone          | Phone number of customer                                            | String   | N/A                     |                                          |                                            |
| Recipient Address        | Address of customer if delivery order, otherwise restaurant address | String   | N/A                     | '13346 Saticoy St.'                      | 74.7% missing and 22.3% restaurant address |
| Recipient Address 2      | Second line of customer or restaurant address                       | String   | N/A                     | 'unit 1'                                 | 97.2% missing                              |
| Recipient Postal Code    | Postal code of recipient or restaurant                              | Number   | N/A                     | 91600                                    | 74.7% missing                              |
| Recipient City           | City of recipient                                                   | String   | N/A                     | 'North Hollywood'                        | 74.7% missing                              |
| Recipient Region         | State of recipient                                                  | String   | N/A                     | 'CA'                                     | Constant                                   |
| Recipient Country        | Country of recipient                                                | String   | N/A                     | 'US'                                     | 71.7% missing and 3% incorrect             |
| Item Quantity            | The quantity of each item in the order                              | Number   | Count                   | 2                                        |                                            |
| Item Name                | Name of item ordered                                                | String   | N/A                     | 'Crispy Chicken Sandwich'                |                                            |
| Item SKU                 | Stock Keeping Unit of item ordered                                  |          |                         |                                          | All missing                                |
| Item Variation           | Subcategory of item ordered                                         | String   | N/A                     | 'Regular'                                |                                            |
| Item Modifiers           | Modifications to items                                              | String   | number x 'Adjustment'   | '1 x Lemon Pepper'                       |                                            |
| Item Price               | Base price of item                                                  | Number   | USD                     | 34.93                                    |                                            |
| Item Options Total Price | Total price of modifications applied                                | Number   | USD                     | 8.49                                     |                                            |
| Item Total Price         | Total price of item with modifications                              | Number   | USD                     | 43.42                                    |                                            |

### Notes
- Each row represents a separate item order. Multiple rows can consist for a single transaction.