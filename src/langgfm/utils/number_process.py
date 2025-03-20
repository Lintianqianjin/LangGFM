# class NumberFormatter:
#     """
#     A class that formats numbers into a specific text format with metadata about digit counts,
#     and can parse the formatted text back to the original number.
    
#     The class normalizes numbers by removing trailing zeros in the decimal part and supports negative numbers.
    
#     Formatted text follows these patterns:
    
#     For positive numbers with decimal parts:
#     <number><digit_info>int_digits.decimal_digits<integer_part>comma_separated_int_digits<decimal_part>comma_separated_decimal_digits</number>
    
#     For positive integers:
#     <number><digit_info>int_digits<integer_part>comma_separated_int_digits</number>
    
#     For negative numbers, a <negative> tag is added after the number tag:
#     <number><negative><digit_info>int_digits.decimal_digits<integer_part>...
#     """
    
#     @staticmethod
#     def format_number(number):
#         """
#         Convert a number to the specified text format
        
#         Args:
#             number: The number to format (integer or float)
            
#         Returns:
#             A formatted string according to the specified pattern
#         """
#         # Check if the number is negative
#         is_negative = number < 0
#         # Work with the absolute value for processing
#         abs_number = abs(number)
        
#         # Convert the number to string
#         str_number = str(abs_number)
        
#         # Split into integer and decimal parts
#         if '.' in str_number:
#             int_part, decimal_part = str_number.split('.')
#             # Normalize by removing trailing zeros in decimal part
#             decimal_part = decimal_part.rstrip('0')
#         else:
#             int_part = str_number
#             decimal_part = ""
        
#         # Count the digits in each part
#         int_digits = len(int_part)
#         decimal_digits = len(decimal_part)
        
#         # Format the parts with commas between digits
#         formatted_int_part = ','.join(int_part)
        
#         # Build the digit info tag differently based on whether there's a decimal part
#         if decimal_part:
#             digit_info = f"{int_digits}.{decimal_digits}"
#         else:
#             digit_info = f"{int_digits}"
        
#         # Start building the result
#         result = "<number>"
        
#         # Add negative tag if needed
#         if is_negative:
#             result += "<negative>"
        
#         # Continue building the result
#         result += f"<digit_info>{digit_info}<integer_part>{formatted_int_part}"
        
#         # Add decimal part only if it exists after normalization
#         if decimal_part:
#             formatted_decimal_part = ','.join(decimal_part)
#             result += f"<decimal_part>{formatted_decimal_part}"
        
#         # Add the end tag
#         result += "</number>"
        
#         return result
    
#     @staticmethod
#     def parse_formatted_text(formatted_text):
#         """
#         Parse the formatted text back to the original number
        
#         Args:
#             formatted_text: The formatted text string
            
#         Returns:
#             The original number (float or integer)
#         """
#         try:
#             # Check if the number is negative
#             is_negative = "<negative>" in formatted_text
            
#             # Extract the digits information
#             digits_pos_start = formatted_text.find("<digit_info>") + len("<digit_info>")
#             digits_pos_end = formatted_text.find("<integer_part>")
            
#             digits_info = formatted_text[digits_pos_start:digits_pos_end]
            
#             # Extract the integer part
#             int_pos_start = digits_pos_end + len("<integer_part>")
            
#             # Check if there's a decimal part
#             if "<decimal_part>" in formatted_text:
#                 int_pos_end = formatted_text.find("<decimal_part>")
#                 int_part_str = formatted_text[int_pos_start:int_pos_end].replace(',', '')
                
#                 # Extract the decimal part
#                 dec_pos_start = int_pos_end + len("<decimal_part>")
#                 dec_pos_end = formatted_text.find("</number>")
#                 decimal_part_str = formatted_text[dec_pos_start:dec_pos_end].replace(',', '')
                
#                 # Rebuild as a float
#                 result = float(f"{int_part_str}.{decimal_part_str}")
#             else:
#                 # No decimal part, extract integer only
#                 int_pos_end = formatted_text.find("</number>")
#                 int_part_str = formatted_text[int_pos_start:int_pos_end].replace(',', '')
#                 result = int(int_part_str)
            
#             # Apply negative sign if needed
#             if is_negative:
#                 result = -result
                
#             return result
            
#         except Exception as e:
#             raise ValueError(f"Failed to parse formatted text: {formatted_text}. Error: {str(e)}")
        
        
class NumberFormatter:
    """
    A simple class that splits numbers into individual characters with a separator,
    preventing tokenizers from treating them as a single entity.
    """
    
    @staticmethod
    def format_number(number, separator=" "):
        """
        Split a number into individual characters with a separator
        
        Args:
            number: The number to split
            separator: The separator to use between digits (default space)
            
        Returns:
            A string with the number's digits separated
        """
        # Convert the number to string
        str_number = str(number)
        
        # Split each character with the separator
        result = separator.join(str_number)
        
        return result
    
    @staticmethod
    def parse_formatted_text(split_string, separator=" "):
        """
        Restore a split number back to its original form
        
        Args:
            split_string: The split string to restore
            separator: The separator used between digits (default space)
            
        Returns:
            The original number (float or integer)
        """
        # Remove all separators
        joined_string = split_string.replace(separator, "")
        
        # Convert to the appropriate type
        if '.' in joined_string:
            return float(joined_string)
        else:
            return int(joined_string)
        
         
        
if __name__ == "__main__":
    # Test the number formatting
    number = 123456.789
    formatted_text = NumberFormatter.format_number(number)
    print(f"Formatted text for number {number}: {formatted_text}")
    
    parsed_number = NumberFormatter.parse_formatted_text(formatted_text)
    print(f"Parsed number from formatted text: {parsed_number}")
    
    # Test integer formatting
    number = 123456
    formatted_text = NumberFormatter.format_number(number)
    print(f"Formatted text for number {number}: {formatted_text}")
    
    parsed_number = NumberFormatter.parse_formatted_text(formatted_text)
    print(f"Parsed number from formatted text: {parsed_number}")
    
    # Test a number with no int part
    number = 0.123
    formatted_text = NumberFormatter.format_number(number)
    print(f"Formatted text for number {number}: {formatted_text}")
    
    parsed_number = NumberFormatter.parse_formatted_text(formatted_text)
    print(f"Parsed number from formatted text: {parsed_number}")
    
    # Test a number with no decimal part
    number = 123.0
    formatted_text = NumberFormatter.format_number(number)
    print(f"Formatted text for number {number}: {formatted_text}")
    
    parsed_number = NumberFormatter.parse_formatted_text(formatted_text)
    print(f"Parsed number from formatted text: {parsed_number}")
    
    # Test a number with no decimal part
    number = -123.01
    formatted_text = NumberFormatter.format_number(number)
    print(f"Formatted text for number {number}: {formatted_text}")
    
    parsed_number = NumberFormatter.parse_formatted_text(formatted_text)
    print(f"Parsed number from formatted text: {parsed_number}")
    