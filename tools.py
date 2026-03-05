"""
Tools module for the multi-agent system.
Defines all tools for both the music catalog and invoice information sub-agents.

All queries use parameterized SQL via run_query_safe().
All tools return named-column dicts for clear LLM interpretation.
All music tools return FULL track details (composer, genre, duration, price, media type).
All tools include structured logging.
"""

import logging
from langchain_core.tools import tool
from database import run_query_safe

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Music Catalog Tools
# ─────────────────────────────────────────────

@tool
def get_albums_by_artist(artist: str) -> str:
    """Get albums by an artist from the music catalog. Uses fuzzy matching on artist name."""
    logger.info(f"TOOL_CALL: get_albums_by_artist | artist={artist}")
    try:
        result = run_query_safe(
            """
            SELECT Album.Title AS AlbumTitle, Artist.Name AS ArtistName
            FROM Album
            JOIN Artist ON Album.ArtistId = Artist.ArtistId
            WHERE Artist.Name LIKE :pattern
            ORDER BY Album.Title;
            """,
            {"pattern": f"%{artist}%"},
        )
        logger.info(f"TOOL_RESULT: get_albums_by_artist | result_length={len(result)}")
        if result == "[]":
            return f"No albums found for artist: {artist}"
        return result
    except Exception as e:
        logger.error(f"Error in get_albums_by_artist: {e}")
        return f"Error looking up albums for '{artist}'. Please try again."


@tool
def get_tracks_by_artist(artist: str) -> str:
    """
    Get songs/tracks by an artist from the catalog.
    Returns total count and a sample of up to 20 tracks with FULL details:
    TrackId, SongName, ArtistName, AlbumTitle, GenreName, Composer,
    Milliseconds, DurationMinutes, Bytes, UnitPrice, MediaType.
    """
    logger.info(f"TOOL_CALL: get_tracks_by_artist | artist={artist}")
    try:
        count_result = run_query_safe(
            """
            SELECT COUNT(*) AS total_tracks
            FROM Track
            JOIN Album ON Track.AlbumId = Album.AlbumId
            JOIN Artist ON Album.ArtistId = Artist.ArtistId
            WHERE Artist.Name LIKE :pattern;
            """,
            {"pattern": f"%{artist}%"},
        )

        result = run_query_safe(
            """
            SELECT Track.TrackId,
                   Track.Name AS SongName,
                   Artist.Name AS ArtistName,
                   Album.Title AS AlbumTitle,
                   Genre.Name AS GenreName,
                   Track.Composer,
                   Track.Milliseconds,
                   ROUND(Track.Milliseconds / 60000.0, 1) AS DurationMinutes,
                   Track.Bytes,
                   Track.UnitPrice,
                   MediaType.Name AS MediaType
            FROM Track
            JOIN Album ON Track.AlbumId = Album.AlbumId
            JOIN Artist ON Album.ArtistId = Artist.ArtistId
            LEFT JOIN Genre ON Track.GenreId = Genre.GenreId
            LEFT JOIN MediaType ON Track.MediaTypeId = MediaType.MediaTypeId
            WHERE Artist.Name LIKE :pattern
            ORDER BY Album.Title, Track.Name
            LIMIT 20;
            """,
            {"pattern": f"%{artist}%"},
        )
        logger.info(f"TOOL_RESULT: get_tracks_by_artist | count={count_result} | sample_length={len(result)}")

        if result == "[]":
            return f"No tracks found for artist: {artist}"

        return f"Total tracks found: {count_result}. Sample (up to 20): {result}"
    except Exception as e:
        logger.error(f"Error in get_tracks_by_artist: {e}")
        return f"Error looking up tracks for '{artist}'. Please try again."


@tool
def get_songs_by_genre(genre: str) -> str:
    """
    Fetch a representative sample of songs from a specific genre.
    Returns total count and a sample (one per artist, up to 10) with FULL details:
    TrackId, SongName, ArtistName, AlbumTitle, GenreName, Composer,
    Milliseconds, DurationMinutes, Bytes, UnitPrice, MediaType.
    """
    logger.info(f"TOOL_CALL: get_songs_by_genre | genre={genre}")
    try:
        count_result = run_query_safe(
            """
            SELECT COUNT(*) AS total_tracks
            FROM Track
            JOIN Genre ON Track.GenreId = Genre.GenreId
            WHERE Genre.Name LIKE :pattern;
            """,
            {"pattern": f"%{genre}%"},
        )

        result = run_query_safe(
            """
            SELECT Track.TrackId,
                   Track.Name AS SongName,
                   Artist.Name AS ArtistName,
                   Album.Title AS AlbumTitle,
                   Genre.Name AS GenreName,
                   Track.Composer,
                   Track.Milliseconds,
                   ROUND(Track.Milliseconds / 60000.0, 1) AS DurationMinutes,
                   Track.Bytes,
                   Track.UnitPrice,
                   MediaType.Name AS MediaType
            FROM Track
            JOIN Genre ON Track.GenreId = Genre.GenreId
            JOIN Album ON Track.AlbumId = Album.AlbumId
            JOIN Artist ON Album.ArtistId = Artist.ArtistId
            LEFT JOIN MediaType ON Track.MediaTypeId = MediaType.MediaTypeId
            WHERE Genre.Name LIKE :pattern
            GROUP BY Artist.Name
            ORDER BY Artist.Name
            LIMIT 10;
            """,
            {"pattern": f"%{genre}%"},
        )
        logger.info(f"TOOL_RESULT: get_songs_by_genre | count={count_result} | sample_length={len(result)}")

        if result == "[]":
            return f"No songs found for the genre: {genre}"

        return (
            f"Total {genre} tracks in catalog: {count_result}. "
            f"Representative sample (one per artist, up to 10): {result}"
        )
    except Exception as e:
        logger.error(f"Error in get_songs_by_genre: {e}")
        return f"Error looking up songs for genre '{genre}'. Please try again."


@tool
def check_for_songs(song_title: str) -> str:
    """
    Check if a song exists in the catalog by its name. Uses fuzzy matching.
    Returns FULL track details: TrackId, SongName, ArtistName, AlbumTitle,
    GenreName, Composer, Milliseconds, DurationMinutes, Bytes, UnitPrice, MediaType.
    """
    logger.info(f"TOOL_CALL: check_for_songs | song_title={song_title}")
    try:
        result = run_query_safe(
            """
            SELECT Track.TrackId,
                   Track.Name AS SongName,
                   Artist.Name AS ArtistName,
                   Album.Title AS AlbumTitle,
                   Genre.Name AS GenreName,
                   Track.Composer,
                   Track.Milliseconds,
                   ROUND(Track.Milliseconds / 60000.0, 1) AS DurationMinutes,
                   Track.Bytes,
                   Track.UnitPrice,
                   MediaType.Name AS MediaType
            FROM Track
            JOIN Album ON Track.AlbumId = Album.AlbumId
            JOIN Artist ON Album.ArtistId = Artist.ArtistId
            LEFT JOIN Genre ON Track.GenreId = Genre.GenreId
            LEFT JOIN MediaType ON Track.MediaTypeId = MediaType.MediaTypeId
            WHERE Track.Name LIKE :pattern
            ORDER BY Track.Name
            LIMIT 10;
            """,
            {"pattern": f"%{song_title}%"},
        )
        logger.info(f"TOOL_RESULT: check_for_songs | result_length={len(result)}")
        if result == "[]":
            return f"No songs found matching: {song_title}"
        return result
    except Exception as e:
        logger.error(f"Error in check_for_songs: {e}")
        return f"Error looking up song '{song_title}'. Please try again."


@tool
def get_track_details(track_id: str) -> str:
    """
    Get complete details for a specific track by its TrackId.
    Returns ALL fields: TrackId, SongName, ArtistName, AlbumTitle, GenreName,
    Composer, Milliseconds, DurationMinutes, Bytes, SizeMB, UnitPrice, MediaType.
    Use this when the customer needs detailed info about a specific known track.
    """
    logger.info(f"TOOL_CALL: get_track_details | track_id={track_id}")
    try:
        result = run_query_safe(
            """
            SELECT Track.TrackId,
                   Track.Name AS SongName,
                   Artist.Name AS ArtistName,
                   Album.Title AS AlbumTitle,
                   Genre.Name AS GenreName,
                   Track.Composer,
                   Track.Milliseconds,
                   ROUND(Track.Milliseconds / 60000.0, 1) AS DurationMinutes,
                   Track.Bytes,
                   ROUND(Track.Bytes / 1048576.0, 1) AS SizeMB,
                   Track.UnitPrice,
                   MediaType.Name AS MediaType
            FROM Track
            LEFT JOIN Album ON Track.AlbumId = Album.AlbumId
            LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
            LEFT JOIN Genre ON Track.GenreId = Genre.GenreId
            LEFT JOIN MediaType ON Track.MediaTypeId = MediaType.MediaTypeId
            WHERE Track.TrackId = :track_id;
            """,
            {"track_id": int(track_id)},
        )
        logger.info(f"TOOL_RESULT: get_track_details | result_length={len(result)}")
        if result == "[]":
            return f"No track found with TrackId: {track_id}"
        return result
    except Exception as e:
        logger.error(f"Error in get_track_details: {e}")
        return f"Error looking up track {track_id}. Please try again."


# ─────────────────────────────────────────────
# Invoice Information Tools
# ─────────────────────────────────────────────

@tool
def get_invoices_by_customer_sorted_by_date(customer_id: str) -> str:
    """
    Look up all invoices for a customer using their ID.
    Returns invoices sorted by date (most recent first) with named columns.
    """
    logger.info(f"TOOL_CALL: get_invoices_by_customer_sorted_by_date | customer_id={customer_id}")
    try:
        result = run_query_safe(
            """
            SELECT InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity,
                   BillingState, BillingCountry, BillingPostalCode, Total
            FROM Invoice
            WHERE CustomerId = :customer_id
            ORDER BY InvoiceDate DESC;
            """,
            {"customer_id": int(customer_id)},
        )
        logger.info(f"TOOL_RESULT: get_invoices_by_customer_sorted_by_date | result_length={len(result)}")
        if result == "[]":
            return f"No invoices found for customer {customer_id}."
        return result
    except Exception as e:
        logger.error(f"Error in get_invoices_by_customer_sorted_by_date: {e}")
        return f"Error retrieving invoices for customer {customer_id}. Please try again."


@tool
def get_invoices_sorted_by_unit_price(customer_id: str) -> str:
    """
    Look up all invoices for a customer, sorted by unit price from highest to lowest.
    Returns invoice details with line item prices and named columns.
    """
    logger.info(f"TOOL_CALL: get_invoices_sorted_by_unit_price | customer_id={customer_id}")
    try:
        result = run_query_safe(
            """
            SELECT Invoice.InvoiceId, Invoice.InvoiceDate, Invoice.Total,
                   InvoiceLine.UnitPrice, InvoiceLine.Quantity
            FROM Invoice
            JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId
            WHERE Invoice.CustomerId = :customer_id
            ORDER BY InvoiceLine.UnitPrice DESC;
            """,
            {"customer_id": int(customer_id)},
        )
        logger.info(f"TOOL_RESULT: get_invoices_sorted_by_unit_price | result_length={len(result)}")
        if result == "[]":
            return f"No invoices found for customer {customer_id}."
        return result
    except Exception as e:
        logger.error(f"Error in get_invoices_sorted_by_unit_price: {e}")
        return f"Error retrieving invoices for customer {customer_id}. Please try again."


@tool
def get_employee_by_invoice_and_customer(invoice_id: str, customer_id: str) -> str:
    """
    Find the employee (support rep) associated with a specific invoice and customer.
    Returns employee full name (first + last), title, and email.
    """
    logger.info(f"TOOL_CALL: get_employee_by_invoice_and_customer | invoice_id={invoice_id}, customer_id={customer_id}")
    try:
        result = run_query_safe(
            """
            SELECT Employee.FirstName, Employee.LastName, Employee.Title, Employee.Email
            FROM Employee
            JOIN Customer ON Customer.SupportRepId = Employee.EmployeeId
            JOIN Invoice ON Invoice.CustomerId = Customer.CustomerId
            WHERE Invoice.InvoiceId = :invoice_id AND Invoice.CustomerId = :customer_id;
            """,
            {"invoice_id": int(invoice_id), "customer_id": int(customer_id)},
        )
        logger.info(f"TOOL_RESULT: get_employee_by_invoice_and_customer | result_length={len(result)}")
        if result == "[]":
            return f"No employee found for invoice ID {invoice_id} and customer ID {customer_id}."
        return result
    except Exception as e:
        logger.error(f"Error in get_employee_by_invoice_and_customer: {e}")
        return f"Error finding employee for invoice {invoice_id}. Please try again."


@tool
def get_invoice_line_items(invoice_id: str, customer_id: str) -> str:
    """
    Get the detailed line items (tracks purchased) for a specific invoice.
    Returns full track details: TrackId, TrackName, ArtistName, AlbumTitle,
    GenreName, Composer, DurationMinutes, UnitPrice, Quantity.
    Use this when the customer asks WHAT they purchased (not just how much).
    """
    logger.info(f"TOOL_CALL: get_invoice_line_items | invoice_id={invoice_id}, customer_id={customer_id}")
    try:
        result = run_query_safe(
            """
            SELECT Track.TrackId,
                   Track.Name AS TrackName,
                   Artist.Name AS ArtistName,
                   Album.Title AS AlbumTitle,
                   Genre.Name AS GenreName,
                   Track.Composer,
                   Track.Milliseconds,
                   ROUND(Track.Milliseconds / 60000.0, 1) AS DurationMinutes,
                   InvoiceLine.UnitPrice,
                   InvoiceLine.Quantity
            FROM InvoiceLine
            JOIN Invoice ON InvoiceLine.InvoiceId = Invoice.InvoiceId
            JOIN Track ON InvoiceLine.TrackId = Track.TrackId
            LEFT JOIN Album ON Track.AlbumId = Album.AlbumId
            LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
            LEFT JOIN Genre ON Track.GenreId = Genre.GenreId
            WHERE Invoice.InvoiceId = :invoice_id AND Invoice.CustomerId = :customer_id
            ORDER BY Track.Name;
            """,
            {"invoice_id": int(invoice_id), "customer_id": int(customer_id)},
        )
        logger.info(f"TOOL_RESULT: get_invoice_line_items | result_length={len(result)}")
        if result == "[]":
            return f"No line items found for invoice {invoice_id} (customer {customer_id})."
        return result
    except Exception as e:
        logger.error(f"Error in get_invoice_line_items: {e}")
        return f"Error retrieving line items for invoice {invoice_id}. Please try again."


# ─────────────────────────────────────────────
# Tool lists for easy access
# ─────────────────────────────────────────────
music_tools = [
    get_albums_by_artist,
    get_tracks_by_artist,
    get_songs_by_genre,
    check_for_songs,
    get_track_details,
]

invoice_tools = [
    get_invoices_by_customer_sorted_by_date,
    get_invoices_sorted_by_unit_price,
    get_employee_by_invoice_and_customer,
    get_invoice_line_items,
]
